"""
Streaming Dataflow pipeline for real-time headway prediction.

Pipeline steps:
  1. Pub/Sub Source     — consume raw GTFS-RT feed snapshots (JSON)
  2. Arrival Detection  — parse feed, diff vehicles, resolve track, emit baseline records
  3. Feature Engineering — apply shared transforms (headway, travel time, track gap, etc.)
  4. Window Buffer      — accumulate 20-observation encoder window per group_id
  5. Sink               — write latest window to Firestore for real-time serving
"""

import argparse
import json
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    GoogleCloudOptions,
    SetupOptions,
    StandardOptions,
)
from google.cloud import storage as gcs_storage

from pipelines.beam.transforms.arrival_detector import DetectArrivalsFn
from pipelines.beam.transforms.transforms import (
    EnrichRecordFn,
    CalculateUpstreamHeadwayFn,
    CalculateUpstreamTravelTimeFn,
    CalculateServiceHeadwayFn,
    CalculateTrackGapFn,
    CalculateTravelTimeDeviationFn,
    EnrichWithEmpiricalFn,
)
from pipelines.beam.transforms.window_buffer import BufferWindowFn
from pipelines.beam.transforms.predict import PredictHeadwayFn
from pipelines.beam.transforms.firestore_sink import WriteToFirestoreFn

logger = logging.getLogger(__name__)

TARGET_STATION = "A32S"

# Default GCS paths for side input maps
_DEFAULT_EMPIRICAL = "gs://realtime-headway-prediction-pipelines/side_inputs/empirical_map.json"
_DEFAULT_MEDIAN_TT = "gs://realtime-headway-prediction-pipelines/side_inputs/median_tt_map.json"


def _load_map_from_gcs(gcs_path: str) -> dict:
    """Download a JSON map from GCS and return as a Python dict."""
    # Parse gs://bucket/path
    path = gcs_path.replace("gs://", "")
    bucket_name = path.split("/", 1)[0]
    blob_path = path.split("/", 1)[1]

    client = gcs_storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    raw = json.loads(blob.download_as_text())
    return raw


def _convert_empirical_keys(raw: dict) -> dict:
    """Convert 'A,Weekday,10' string keys to ('A','Weekday',10) tuple keys."""
    out = {}
    for k, v in raw.items():
        parts = k.split(",")
        out[(parts[0], parts[1], int(parts[2]))] = v
    return out


def _convert_median_tt_keys(raw: dict) -> dict:
    """Convert 'A,Weekday,10,A31S' string keys to ('A','Weekday',10,'A31S') tuple keys."""
    out = {}
    for k, v in raw.items():
        parts = k.split(",")
        out[(parts[0], parts[1], int(parts[2]), parts[3])] = v
    return out

FEATURE_FIELDS = [
    "service_headway",
    "upstream_headway_14th",
    "travel_time_14th",
    "travel_time_23rd",
    "travel_time_34th",
    "preceding_train_gap",
    "travel_time_14th_deviation",
    "travel_time_23rd_deviation",
    "travel_time_34th_deviation",
    "empirical_median",
    "hour_sin",
    "hour_cos",
    "time_idx",
    "regime_id",
]


def _add_stops_at_23rd(element):
    """Derive stops_at_23rd binary flag from travel_time_23rd."""
    tt23 = element.get("travel_time_23rd")
    element["stops_at_23rd"] = 1.0 if (tt23 is not None and tt23 > 0) else 0.0
    return element


def _log_target_station(element):
    """Log every event at the target station with all generated features."""
    if element.get("stop_id") != TARGET_STATION:
        return element
    gid = element.get("group_id", "?")
    route = element.get("route_id", "?")
    track = element.get("track_id", "?")
    trip = element.get("trip_uid", "?")
    at = element.get("arrival_time", "?")
    features = {k: element.get(k) for k in FEATURE_FIELDS}
    logger.info(
        "\n╔══ TARGET STATION EVENT ══════════════════════════"
        "\n║ group: %-10s route: %s  track: %s"
        "\n║ trip:  %s"
        "\n║ arrival_time: %s"
        "\n║ features:"
        "\n║   service_headway:    %s"
        "\n║   upstream_hw_14th:   %s"
        "\n║   travel_time_14th:   %s"
        "\n║   travel_time_23rd:   %s"
        "\n║   travel_time_34th:   %s"
        "\n║   preceding_gap:      %s"
        "\n║   preceding_route:    %s"
        "\n║   tt_dev_14th:        %s"
        "\n║   tt_dev_23rd:        %s"
        "\n║   tt_dev_34th:        %s"
        "\n║   empirical_median:   %s"
        "\n║   hour_sin/cos:       %s / %s"
        "\n║   time_idx:           %s"
        "\n║   regime_id:          %s"
        "\n╚══════════════════════════════════════════════════",
        gid, route, track, trip, at,
        features["service_headway"],
        features["upstream_headway_14th"],
        features["travel_time_14th"],
        features["travel_time_23rd"],
        features["travel_time_34th"],
        features["preceding_train_gap"],
        element.get("preceding_route_id", "?"),
        features["travel_time_14th_deviation"],
        features["travel_time_23rd_deviation"],
        features["travel_time_34th_deviation"],
        features["empirical_median"],
        features["hour_sin"], features["hour_cos"],
        features["time_idx"],
        features["regime_id"],
    )
    return element


def run(argv=None):
    parser = argparse.ArgumentParser(
        description="Streaming headway prediction pipeline"
    )
    parser.add_argument(
        "--input_subscription",
        required=True,
        help="Pub/Sub subscription (projects/PROJECT/subscriptions/SUB)",
    )
    parser.add_argument(
        "--project_id",
        default="realtime-headway-prediction",
        help="GCP project ID",
    )
    parser.add_argument(
        "--region",
        default="us-east1",
        help="GCP region for Dataflow",
    )
    parser.add_argument(
        "--empirical_map_gcs",
        default=_DEFAULT_EMPIRICAL,
        help="GCS path to empirical_map.json",
    )
    parser.add_argument(
        "--median_tt_map_gcs",
        default=_DEFAULT_MEDIAN_TT,
        help="GCS path to median_tt_map.json",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)

    # --- Load side input maps from GCS ---
    logger.info("Loading empirical_map from %s", known_args.empirical_map_gcs)
    empirical_map = _convert_empirical_keys(
        _load_map_from_gcs(known_args.empirical_map_gcs)
    )
    logger.info("Loaded empirical_map: %d entries", len(empirical_map))

    logger.info("Loading median_tt_map from %s", known_args.median_tt_map_gcs)
    median_tt_map = _convert_median_tt_keys(
        _load_map_from_gcs(known_args.median_tt_map_gcs)
    )
    logger.info("Loaded median_tt_map: %d entries", len(median_tt_map))

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(StandardOptions).streaming = True

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = known_args.project_id
    google_cloud_options.region = known_args.region

    with beam.Pipeline(options=pipeline_options) as p:

        # --- Step 1: Pub/Sub Source ---
        raw_messages = (
            p
            | "ReadPubSub" >> beam.io.ReadFromPubSub(
                subscription=known_args.input_subscription,
            )
        )

        # --- Step 2: Arrival Detection ---
        arrivals = (
            raw_messages
            | "KeyForState" >> beam.Map(lambda msg: ("all_routes", msg))
            | "DetectArrivals" >> beam.ParDo(DetectArrivalsFn())
        )

        # --- Step 3: Feature Engineering ---
        # 3a. Row-level enrichment (flat — adds time_idx, group_id, regime, etc.)
        enriched = (
            arrivals
            | "EnrichRecords" >> beam.ParDo(EnrichRecordFn())
        )

        # 3b. Upstream headway at 14th St (keyed by group_id)
        with_upstream_hw = (
            enriched
            | "KeyByGroupUpstream" >> beam.Map(
                lambda x: (x.get("group_id", "Unknown"), x)
            )
            | "CalcUpstreamHeadway" >> beam.ParDo(CalculateUpstreamHeadwayFn())
        )

        # 3c. Upstream travel time (keyed by trip_uid — filters to target station)
        with_travel_time = (
            with_upstream_hw
            | "KeyByTrip" >> beam.Map(lambda x: (x["trip_uid"], x))
            | "CalcTravelTime" >> beam.ParDo(CalculateUpstreamTravelTimeFn())
        )

        # 3d. Service headway — target (keyed by group_id)
        with_headway = (
            with_travel_time
            | "KeyByGroupHeadway" >> beam.Map(lambda x: (x["group_id"], x))
            | "CalcServiceHeadway" >> beam.ParDo(CalculateServiceHeadwayFn())
            | "FilterInvalidHeadways" >> beam.Filter(
                lambda x: x.get("service_headway") is None
                or (0.5 < x["service_headway"] < 120)
            )
        )

        # 3e. Track gap (keyed by track_id)
        with_track_gap = (
            with_headway
            | "KeyByTrack" >> beam.Map(lambda x: (x["track_id"], x))
            | "CalcTrackGap" >> beam.ParDo(CalculateTrackGapFn())
        )

        # 3f. Travel time deviation (flat, with side input)
        with_deviation = (
            with_track_gap
            | "CalcDeviation" >> beam.ParDo(
                CalculateTravelTimeDeviationFn(),
                median_tt_map=median_tt_map,
            )
        )

        # 3g. Empirical headway (flat, with side input)
        feature_records = (
            with_deviation
            | "EnrichEmpirical" >> beam.ParDo(
                EnrichWithEmpiricalFn(),
                empirical_map=empirical_map,
            )
        )

        # Feedback: log every target station event with all features
        feature_records = feature_records | "LogTargetStation" >> beam.Map(
            _log_target_station
        )

        # 3h. Derived binary flag: does this train stop at 23rd St?
        feature_records = feature_records | "AddStopsAt23rd" >> beam.Map(
            _add_stops_at_23rd
        )

        # --- Step 4: Window Buffer ---
        windows = (
            feature_records
            | "KeyByGroupBuffer" >> beam.Map(
                lambda x: (x.get("group_id", "Unknown"), x)
            )
            | "BufferWindow" >> beam.ParDo(BufferWindowFn())
        )

        # --- Step 5: Prediction ---
        predictions = (
            windows
            | "Predict" >> beam.ParDo(
                PredictHeadwayFn(project=known_args.project_id)
            )
        )

        # --- Step 6: Sink ---
        _ = (
            predictions
            | "WriteFirestore" >> beam.ParDo(
                WriteToFirestoreFn(project=known_args.project_id)
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()

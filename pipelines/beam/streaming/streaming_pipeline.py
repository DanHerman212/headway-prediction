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
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    GoogleCloudOptions,
    SetupOptions,
    StandardOptions,
)

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
from pipelines.beam.transforms.firestore_sink import WriteToFirestoreFn

logger = logging.getLogger(__name__)


def _log_tap(label):
    """Return a logging function that logs and passes through the element."""
    def _log(element):
        if isinstance(element, dict):
            gid = element.get("group_id", "?")
            logger.info("[%s] group_id=%s | keys=%s", label, gid, sorted(element.keys()))
        else:
            logger.info("[%s] %s", label, str(element)[:200])
        return element
    return _log


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

    known_args, pipeline_args = parser.parse_known_args(argv)

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

        # Tap: arrival detection output
        arrivals = arrivals | "TapArrivals" >> beam.Map(_log_tap("ARRIVAL"))

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
                median_tt_map={},  # TODO: load from GCS
            )
        )

        # 3g. Empirical headway (flat, with side input)
        feature_records = (
            with_deviation
            | "EnrichEmpirical" >> beam.ParDo(
                EnrichWithEmpiricalFn(),
                empirical_map={},  # TODO: load from GCS
            )
        )

        # Tap: fully-engineered records
        feature_records = feature_records | "TapFeatures" >> beam.Map(
            _log_tap("FEATURES")
        )

        # --- Step 4: Window Buffer ---
        windows = (
            feature_records
            | "KeyByGroupBuffer" >> beam.Map(
                lambda x: (x.get("group_id", "Unknown"), x)
            )
            | "BufferWindow" >> beam.ParDo(BufferWindowFn())
        )

        # Tap: windows emitted
        windows = windows | "TapWindows" >> beam.Map(
            lambda w: (
                logger.info(
                    "[WINDOW] group_id=%s  obs_count=%d",
                    w.get("group_id", "?"),
                    len(w.get("observations", [])),
                ),
                w,
            )[-1]
        )

        # --- Step 5: Sink ---
        _ = (
            windows
            | "WriteFirestore" >> beam.ParDo(
                WriteToFirestoreFn(project=known_args.project_id)
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()

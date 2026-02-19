"""
BigQuery sinks for streaming headway prediction monitoring.

Two tables track prediction quality:
  actuals     — ground-truth headways from detected arrivals
  predictions — model forecasts (P10/P50/P90) from the endpoint

A scheduled BQ query joins these to compute MAE/sMAPE hourly.
Tables must exist before the pipeline starts — run:
    python scripts/setup_bq_monitoring.py
"""

import logging
from datetime import datetime, timezone

import apache_beam as beam

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Schemas (dict format for beam.io.WriteToBigQuery)
# --------------------------------------------------------------------------- #

ACTUALS_SCHEMA = {
    "fields": [
        {"name": "group_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "time_idx", "type": "INTEGER", "mode": "REQUIRED"},
        {"name": "service_headway", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "arrival_time", "type": "STRING", "mode": "NULLABLE"},
        {"name": "route_id", "type": "STRING", "mode": "NULLABLE"},
        {"name": "stop_id", "type": "STRING", "mode": "NULLABLE"},
        {"name": "recorded_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}

PREDICTIONS_SCHEMA = {
    "fields": [
        {"name": "group_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "last_observation_time_idx", "type": "INTEGER", "mode": "REQUIRED"},
        {"name": "headway_p10", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "headway_p50", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "headway_p90", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "predicted_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}


# --------------------------------------------------------------------------- #
#  Row formatters
# --------------------------------------------------------------------------- #

def _format_actual(element):
    """Map a feature-engineered record to an actuals BQ row."""
    return {
        "group_id": element["group_id"],
        "time_idx": element["time_idx"],
        "service_headway": element.get("service_headway"),
        "arrival_time": element.get("arrival_time"),
        "route_id": element.get("route_id"),
        "stop_id": element.get("stop_id"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _format_prediction(element):
    """Map a prediction result to a predictions BQ row."""
    observations = element.get("observations", [])
    last_time_idx = observations[-1]["time_idx"] if observations else None

    return {
        "group_id": element["group_id"],
        "last_observation_time_idx": last_time_idx,
        "headway_p10": element.get("headway_p10"),
        "headway_p50": element.get("headway_p50"),
        "headway_p90": element.get("headway_p90"),
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
#  Composite PTransforms
# --------------------------------------------------------------------------- #

class WriteActuals(beam.PTransform):
    """Write ground-truth arrival headways to BigQuery.

    Expects feature-engineered dicts with at minimum:
      group_id, time_idx, service_headway
    """

    def __init__(self, project: str, dataset: str = "headway_monitoring"):
        super().__init__()
        self._table = f"{project}:{dataset}.actuals"

    def expand(self, pcoll):
        return (
            pcoll
            | "FormatActualRow" >> beam.Map(_format_actual)
            | "InsertActuals" >> beam.io.WriteToBigQuery(
                table=self._table,
                schema=ACTUALS_SCHEMA,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
                method=beam.io.WriteToBigQuery.Method.STREAMING_INSERTS,
            )
        )


class WritePredictions(beam.PTransform):
    """Write model predictions to BigQuery.

    Expects prediction dicts with:
      group_id, observations (list), headway_p10/p50/p90
    """

    def __init__(self, project: str, dataset: str = "headway_monitoring"):
        super().__init__()
        self._table = f"{project}:{dataset}.predictions"

    def expand(self, pcoll):
        return (
            pcoll
            | "FilterHasTimeIdx" >> beam.Filter(
                lambda x: (
                    x.get("observations")
                    and x["observations"][-1].get("time_idx") is not None
                )
            )
            | "FormatPredictionRow" >> beam.Map(_format_prediction)
            | "InsertPredictions" >> beam.io.WriteToBigQuery(
                table=self._table,
                schema=PREDICTIONS_SCHEMA,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
                method=beam.io.WriteToBigQuery.Method.STREAMING_INSERTS,
            )
        )

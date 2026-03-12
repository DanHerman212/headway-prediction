"""
Graph WaveNet dataset generation -- streaming Beam pipeline.

Reads dense-grid snapshots from Pub/Sub, computes the backward feature
(minutes_since_last_train) and forward target (time_to_next_train) via
a stateful buffer-then-backfill pattern, and writes ML-ready rows to
BigQuery.

Architecture:
  Pub/Sub --> Explode 137-row snapshot --> Key by node_id
          --> LabelTimeToNextTrainFn (stateful) --> BigQuery

Usage (Dataflow):
  python -m pipelines.graph_wavenet.pipeline \
      --input_subscription projects/PROJECT/subscriptions/SUB \
      --output_table PROJECT:graph_wavenet.dense_grid_labeled \
      --runner DataflowRunner \
      --project PROJECT \
      --region us-east1 \
      --temp_location gs://BUCKET/tmp \
      --staging_location gs://BUCKET/staging

Usage (local / DirectRunner):
  python -m pipelines.graph_wavenet.pipeline \
      --input_file local_artifacts/dense_grid.csv \
      --output_table "" \
      --runner DirectRunner
"""

import argparse
import csv
import json
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
)

from pipelines.graph_wavenet.transforms import LabelTimeToNextTrainFn

logger = logging.getLogger(__name__)

# BigQuery schema for the labeled output table
BQ_SCHEMA = (
    "snapshot_time:DATETIME,"
    "node_id:STRING,"
    "train_present:INTEGER,"
    "route_id:STRING,"
    "trip_id:STRING,"
    "minutes_since_last_train:FLOAT,"
    "time_to_next_train:FLOAT"
)


def _explode_pubsub(message):
    """Unpack a single Pub/Sub message (JSON array) into 137 rows."""
    rows = json.loads(message.decode("utf-8"))
    for row in rows:
        yield row


def _key_by_node(row):
    """Key each row by node_id for stateful processing."""
    return (row["node_id"], row)


def _read_csv_rows(pipeline, csv_path):
    """Read a local CSV file and yield dicts (for DirectRunner testing)."""

    def _parse_csv(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["train_present"] = int(row["train_present"])
                yield row

    return (
        pipeline
        | "ReadCSVPath" >> beam.Create([csv_path])
        | "ParseCSV" >> beam.FlatMap(_parse_csv)
    )


def run(argv=None):
    parser = argparse.ArgumentParser(
        description="Graph WaveNet dense-grid labeling pipeline"
    )
    parser.add_argument(
        "--input_subscription",
        default="",
        help="Pub/Sub subscription (projects/P/subscriptions/S). "
             "Leave empty for local CSV mode.",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help="Path to local CSV (DirectRunner testing only).",
    )
    parser.add_argument(
        "--output_table",
        default="",
        help="BigQuery table (PROJECT:DATASET.TABLE). "
             "Leave empty to print to stdout.",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="Local CSV output path (DirectRunner testing only).",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)

    options = PipelineOptions(pipeline_args)
    options.view_as(SetupOptions).save_main_session = True

    # Streaming mode only when reading from Pub/Sub
    if known_args.input_subscription:
        options.view_as(StandardOptions).streaming = True

    with beam.Pipeline(options=options) as p:

        # -- Source ---------------------------------------------------------
        if known_args.input_subscription:
            rows = (
                p
                | "ReadPubSub" >> beam.io.ReadFromPubSub(
                    subscription=known_args.input_subscription,
                )
                | "Explode" >> beam.FlatMap(_explode_pubsub)
            )
        elif known_args.input_file:
            rows = _read_csv_rows(p, known_args.input_file)
        else:
            raise ValueError("Provide --input_subscription or --input_file")

        # -- Label ----------------------------------------------------------
        labeled = (
            rows
            | "KeyByNode" >> beam.Map(_key_by_node)
            | "Label" >> beam.ParDo(LabelTimeToNextTrainFn())
        )

        # -- Sink -----------------------------------------------------------
        if known_args.output_table:
            _ = (
                labeled
                | "WriteBQ" >> beam.io.WriteToBigQuery(
                    known_args.output_table,
                    schema=BQ_SCHEMA,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    method=beam.io.WriteToBigQuery.Method.STREAMING_INSERTS,
                )
            )
        elif known_args.output_file:
            _ = (
                labeled
                | "FormatCSV" >> beam.Map(
                    lambda r: ",".join(str(r.get(c, "")) for c in [
                        "snapshot_time", "node_id", "train_present",
                        "route_id", "trip_id",
                        "minutes_since_last_train", "time_to_next_train",
                    ])
                )
                | "WriteFile" >> beam.io.WriteToText(
                    known_args.output_file,
                    file_name_suffix=".csv",
                    header="snapshot_time,node_id,train_present,route_id,"
                           "trip_id,minutes_since_last_train,time_to_next_train",
                )
            )
        else:
            _ = labeled | "Log" >> beam.Map(
                lambda r: logger.info(
                    "%s | %s | present=%d | since_last=%s | to_next=%s",
                    r["snapshot_time"], r["node_id"], r["train_present"],
                    r.get("minutes_since_last_train"),
                    r.get("time_to_next_train"),
                )
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run()

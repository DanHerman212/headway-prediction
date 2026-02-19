#!/usr/bin/env python3
"""
Set up BigQuery resources for headway prediction error monitoring.

Creates:
  1. Dataset: headway_monitoring (us-east1)
  2. Table:   actuals            — ground-truth headways from arrival detection
  3. Table:   predictions        — P10/P50/P90 forecasts from model endpoint
  4. Table:   evaluation_results — hourly MAE/sMAPE (written by scheduled query)

After running this script, schedule the evaluation query:
  - The script prints the bq CLI command at the end
  - Or schedule it manually in the BigQuery Console → Scheduled Queries

Prerequisites:
    pip install google-cloud-bigquery

Usage:
    python scripts/setup_bq_monitoring.py
    python scripts/setup_bq_monitoring.py --project my-project --dataset my_dataset
"""

import argparse
import logging
import textwrap

from google.cloud import bigquery

logger = logging.getLogger(__name__)

PROJECT = "realtime-headway-prediction"
DATASET = "headway_monitoring"
LOCATION = "us-east1"


# --------------------------------------------------------------------------- #
#  Table schemas
# --------------------------------------------------------------------------- #

ACTUALS_SCHEMA = [
    bigquery.SchemaField("group_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("time_idx", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("service_headway", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("arrival_time", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("route_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("stop_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("recorded_at", "TIMESTAMP", mode="REQUIRED"),
]

PREDICTIONS_SCHEMA = [
    bigquery.SchemaField("group_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("last_observation_time_idx", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("headway_p10", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("headway_p50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("headway_p90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("predicted_at", "TIMESTAMP", mode="REQUIRED"),
]

EVALUATION_SCHEMA = [
    bigquery.SchemaField("eval_time", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("window_hours", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("n_predictions", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("mae_seconds", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("smape_pct", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("median_abs_error", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("p90_abs_error", "FLOAT", mode="NULLABLE"),
]


# --------------------------------------------------------------------------- #
#  Evaluation SQL
# --------------------------------------------------------------------------- #

def _evaluation_sql(project: str, dataset: str) -> str:
    """Return the hourly evaluation query that joins predictions to actuals."""
    return textwrap.dedent(f"""\
        INSERT INTO `{project}.{dataset}.evaluation_results`
          (eval_time, window_hours, n_predictions,
           mae_seconds, smape_pct, median_abs_error, p90_abs_error)

        WITH matched AS (
          SELECT
            p.group_id,
            p.headway_p50,
            a.service_headway   AS actual,
            ABS(a.service_headway - p.headway_p50)  AS abs_error,
            SAFE_DIVIDE(
              ABS(a.service_headway - p.headway_p50),
              (ABS(a.service_headway) + ABS(p.headway_p50)) / 2
            ) AS smape
          FROM `{project}.{dataset}.predictions` p
          INNER JOIN `{project}.{dataset}.actuals` a
            ON  a.group_id = p.group_id
            AND a.time_idx = p.last_observation_time_idx + 1
          WHERE p.headway_p50 IS NOT NULL
            AND a.service_headway IS NOT NULL
            AND a.service_headway > 0
            -- 15-min lag so actuals have time to land
            AND p.predicted_at BETWEEN
                TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 75 MINUTE)
                AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 15 MINUTE)
        )

        SELECT
          CURRENT_TIMESTAMP()  AS eval_time,
          1                    AS window_hours,
          COUNT(*)             AS n_predictions,
          ROUND(AVG(abs_error), 2)                                    AS mae_seconds,
          ROUND(AVG(smape) * 100, 2)                                  AS smape_pct,
          ROUND(APPROX_QUANTILES(abs_error, 100)[OFFSET(50)], 2)     AS median_abs_error,
          ROUND(APPROX_QUANTILES(abs_error, 100)[OFFSET(90)], 2)     AS p90_abs_error
        FROM matched
        HAVING COUNT(*) > 0
    """)


# --------------------------------------------------------------------------- #
#  Setup functions
# --------------------------------------------------------------------------- #

def _create_dataset(client: bigquery.Client, project: str, dataset: str):
    dataset_ref = bigquery.Dataset(f"{project}.{dataset}")
    dataset_ref.location = LOCATION
    ds = client.create_dataset(dataset_ref, exists_ok=True)
    logger.info("Dataset ready: %s.%s (%s)", project, dataset, ds.location)


def _create_table(
    client: bigquery.Client,
    project: str,
    dataset: str,
    table_id: str,
    schema: list,
    partition_field: str | None = None,
):
    table_ref = f"{project}.{dataset}.{table_id}"
    table = bigquery.Table(table_ref, schema=schema)

    if partition_field:
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )

    table = client.create_table(table, exists_ok=True)
    part_info = f" (partitioned by {partition_field})" if partition_field else ""
    logger.info("Table ready: %s%s", table_ref, part_info)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Set up BQ prediction monitoring")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    client = bigquery.Client(project=args.project)

    # 1. Dataset
    _create_dataset(client, args.project, args.dataset)

    # 2. Tables
    _create_table(client, args.project, args.dataset,
                  "actuals", ACTUALS_SCHEMA, partition_field="recorded_at")
    _create_table(client, args.project, args.dataset,
                  "predictions", PREDICTIONS_SCHEMA, partition_field="predicted_at")
    _create_table(client, args.project, args.dataset,
                  "evaluation_results", EVALUATION_SCHEMA, partition_field="eval_time")

    # 3. Print evaluation SQL
    sql = _evaluation_sql(args.project, args.dataset)
    print("\n" + "=" * 70)
    print("EVALUATION QUERY (runs hourly, joins predictions ↔ actuals):")
    print("=" * 70)
    print(sql)

    # 4. Print scheduling command
    # Escape single quotes in SQL for shell
    escaped_sql = sql.replace("'", "'\\''").replace("\n", " ")
    print("=" * 70)
    print("TO SCHEDULE THIS QUERY (hourly):")
    print("=" * 70)
    print(textwrap.dedent(f"""\
        bq mk \\
          --transfer_config \\
          --project_id={args.project} \\
          --data_source=scheduled_query \\
          --target_dataset={args.dataset} \\
          --display_name="Headway Prediction Evaluation (Hourly)" \\
          --location={LOCATION} \\
          --schedule="every 1 hours" \\
          --params='{{"query":"{escaped_sql}","write_disposition":"WRITE_APPEND"}}'
    """))
    print("Or schedule it in the BigQuery Console → Scheduled Queries → Create.")
    print("=" * 70)


if __name__ == "__main__":
    main()

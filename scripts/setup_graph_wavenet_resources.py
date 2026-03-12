"""
Create GCP resources for the Graph WaveNet dataset pipeline.

  1. Pub/Sub topic  + subscription
  2. BigQuery dataset + table (with labeled schema)

Usage:
    python scripts/setup_graph_wavenet_resources.py
    python scripts/setup_graph_wavenet_resources.py --project my-project
"""

import argparse


def setup_pubsub(project, topic_name, sub_name):
    from google.cloud import pubsub_v1
    from google.api_core.exceptions import AlreadyExists

    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    topic_path = publisher.topic_path(project, topic_name)
    sub_path = subscriber.subscription_path(project, sub_name)

    try:
        publisher.create_topic(request={"name": topic_path})
        print(f"Created topic: {topic_path}")
    except AlreadyExists:
        print(f"Topic exists:  {topic_path}")

    try:
        subscriber.create_subscription(
            request={"name": sub_path, "topic": topic_path}
        )
        print(f"Created subscription: {sub_path}")
    except AlreadyExists:
        print(f"Subscription exists:  {sub_path}")

    return topic_path, sub_path


def setup_bigquery(project, dataset_name, table_name):
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    dataset_ref = f"{project}.{dataset_name}"
    table_id = f"{dataset_ref}.{table_name}"

    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset ready: {dataset_ref}")

    schema = [
        bigquery.SchemaField("snapshot_time", "DATETIME", mode="REQUIRED"),
        bigquery.SchemaField("node_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("train_present", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("route_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("trip_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("minutes_since_last_train", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("time_to_next_train", "FLOAT", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="snapshot_time",
    )
    table.clustering_fields = ["node_id"]

    table = client.create_table(table, exists_ok=True)
    print(f"Table ready:   {table_id}")
    print(f"  Partitioned by snapshot_time (DAY), clustered by node_id")
    return table_id


def main():
    parser = argparse.ArgumentParser(
        description="Create GCP resources for Graph WaveNet pipeline"
    )
    parser.add_argument("--project", default="realtime-headway-prediction")
    parser.add_argument("--topic", default="graph-wavenet-snapshots")
    parser.add_argument("--subscription", default="graph-wavenet-snapshots-dataflow")
    parser.add_argument("--bq-dataset", default="graph_wavenet")
    parser.add_argument("--bq-table", default="dense_grid_labeled")
    args = parser.parse_args()

    print("=== Pub/Sub ===")
    topic, sub = setup_pubsub(args.project, args.topic, args.subscription)

    print("\n=== BigQuery ===")
    table = setup_bigquery(args.project, args.bq_dataset, args.bq_table)

    print("\n=== Done ===")
    print(f"Publisher topic:        {topic}")
    print(f"Dataflow subscription:  {sub}")
    print(f"BigQuery table:         {table}")
    print(f"\nTo run the publisher:")
    print(f"  python -m realtime_ingestion.graph_ingestion.grid_publisher \\")
    print(f"      --project {args.project} --topic {args.topic}")
    print(f"\nTo run the Dataflow pipeline:")
    print(f"  python -m pipelines.graph_wavenet.pipeline \\")
    print(f"      --input_subscription {sub} \\")
    print(f"      --output_table {args.project}:{args.bq_dataset}.{args.bq_table} \\")
    print(f"      --runner DataflowRunner --project {args.project} --region us-east1 \\")
    print(f"      --temp_location gs://BUCKET/tmp --staging_location gs://BUCKET/staging")


if __name__ == "__main__":
    main()

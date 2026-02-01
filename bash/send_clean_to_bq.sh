envsubst < batch_ingestion/sql/01_create_clean_table.sql | bq query --use_legacy_sql=false \
  --destination_table=${BQ_DATASET}.clean \
  --location=us-east1 \
  --replace
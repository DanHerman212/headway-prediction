envsubst < batch_ingestion/sql/02_feature_engineering.sql | bq query --use_legacy_sql=false \
  --destination_table=${BQ_DATASET}.ml \
  --location=us-east1 \
  --replace
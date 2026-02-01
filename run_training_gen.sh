#!/bin/bash

# load variables from .env
if [ -f .env ]; then
  # 'set -a' automatically exports all variables defined in the source
  set -a
  source .env
  set +a
else
  echo "Error: .env file not found in current directory. Please run from project root."
  exit 1
fi

# 2 validation
if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCS_BUCKET" ] || [ -z "$GCP_REGION" ]; then
  echo "Error: .env is missing of the: GCP_PROJECT_ID, GCS_BUCKET, GCP_REGION"
  exit 1
fi

JOB_NAME="headway-training-gen-$(date +%Y%m%d-%H%M%S)"
echo "Launcing Dataflow Job: $JOB_NAME"
echo "Project: $GCP_PROJECT_ID"
echo "Bucket: $GCS_BUCKET"

# 3 run pipeline 
# note ensure your .env variable names match strictly
python -m pipelines.beam.batch.generate_dataset \
  --project_id $GCP_PROJECT_ID \
  --temp_location gs://$GCS_BUCKET/temp/ \
  --runner DataflowRunner \
  --region $GCP_REGION \
  --job_name $JOB_NAME \
  --setup_file ./setup.py



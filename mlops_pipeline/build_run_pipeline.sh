#!/bin/bash
set -e

# Change to the directory of the script
cd "$(dirname "$0")"

# Default settings
SKIP_BUILD=false
SUBMIT_JOB=true

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--skip-build) SKIP_BUILD=true ;;
        --dry-run) SUBMIT_JOB=false ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Load environment variables
if [ -f .env ]; then
    echo "Loading configuration from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found. Please create one based on .env.example"
    exit 1
fi

IMAGE_URI=${TENSORFLOW_IMAGE_URI:-"us-docker.pkg.dev/headway-prediction/ml-pipelines/headway-training:latest"}
# pipeline.py uses config("PIPELINE_ROOT"), ensure it matches or defaulted here for the runner
PIPELINE_ROOT=${PIPELINE_ROOT:-"gs://${BUCKET_NAME}/pipeline_root"}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-"your-service-account@${PROJECT_ID}.iam.gserviceaccount.com"}

echo "--------------------------------------------------------"
echo "Configuration:"
echo "Project:   $PROJECT_ID"
echo "Region:    $REGION"
echo "Image URI: $IMAGE_URI"
echo "Run Mode:  Skip Build=$SKIP_BUILD, Submit Job=$SUBMIT_JOB"
echo "--------------------------------------------------------"

# Check for local KFP
if ! python -c "import kfp" &> /dev/null; then
    echo "Warning: KFP SDK not found. Installing..."
    pip install kfp python-dotenv
fi

# 1. Build Docker Image (Conditional)
if [ "$SKIP_BUILD" = true ]; then
    echo "Skipping Docker build and push as requested."
else
    echo "Submitting build to Cloud Build..."
    # Uses .gcloudignore to filter uploaded context
    gcloud builds submit . --tag $IMAGE_URI
fi

# 2. Compile Pipeline
echo "Compiling KFP pipeline..."
python pipeline.py

echo "Pipeline compiled to 'headway_pipeline.json'."

# 3. Submit to Vertex AI (Conditional)
if [ "$SUBMIT_JOB" = true ]; then
    echo "Submitting pipeline job to Vertex AI..."

    # Capture the output to get the name/ID, but allow stdout to show progress
    RUN_ID=$(gcloud beta ai pipelines run \
      --pipeline-file="headway_pipeline.json" \
      --display-name="headway-training-$(date +%Y%m%d-%H%M%S)" \
      --region="$REGION" \
      --project="$PROJECT_ID" \
      --service-account="$SERVICE_ACCOUNT" \
      --pipeline-root="$PIPELINE_ROOT" \
      --format="value(name)")
    
    # Run name format: projects/123/locations/region/pipelineJobs/pipeline-job-id
    # We strip to just ID for link convenience usually, or just use the full link.
    JOB_ID=$(basename $RUN_ID)

    echo ""
    echo "✅ Job Submitted Successfully!"
    echo "Job ID: $JOB_ID"
    echo "Link:   https://console.cloud.google.com/vertex-ai/locations/$REGION/pipelines/runs/$JOB_ID?project=$PROJECT_ID"
else
    echo ""
    echo "Dry run complete. Use normal execution or remove --dry-run to submit to Vertex AI."
fi

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

# Map .env variables to script variables if needed
PROJECT_ID=${PROJECT_ID:-$GCP_PROJECT_ID}
REGION=${REGION:-$VERTEX_LOCATION}
BUCKET_NAME=${BUCKET_NAME:-$GCS_BUCKET_NAME}

# Check if TENSORFLOW_IMAGE_URI is already set in .env or environment
if [ -n "$TENSORFLOW_IMAGE_URI" ]; then
    echo "Using configured TENSORFLOW_IMAGE_URI: $TENSORFLOW_IMAGE_URI"
    IMAGE_URI="$TENSORFLOW_IMAGE_URI"
    # Extract tag if possible for logging, though not strictly needed logic-wise
    IMAGE_TAG=$(echo "$IMAGE_URI" | cut -d':' -f2)
else
    # Generate a unique tag for this build execution to bypass Vertex AI caching
    if [ "$SKIP_BUILD" = true ]; then
        echo "Retrieving latest image tag from Artifact Registry..."
        # Fetch the most recently updated tag
        # Uses gcloud artifacts docker tags list, sorted reverse by update time, limit 1
        LATEST_TAG=$(gcloud artifacts docker tags list "us-docker.pkg.dev/${PROJECT_ID}/headway-pipelines/training" \
            --sort-by=~UPDATE_TIME \
            --limit=1 \
            --format="value(tag)")
            
        if [ -z "$LATEST_TAG" ]; then
            echo "Error: No existing image tags found. Cannot skip build."
            exit 1
        fi
        
        echo "Found latest tag: $LATEST_TAG"
        IMAGE_TAG="$LATEST_TAG"
    else
        TIMESTAMP=$(date +%s)
        IMAGE_TAG="v${TIMESTAMP}"
    fi

    # Base Image URI (Hardcoded pattern to avoid :latest duplication from .env)
    # We force the base URI to be clean, ignoring potentially malformed .env values for this specific build
    BASE_IMAGE_URI="us-docker.pkg.dev/${PROJECT_ID}/headway-pipelines/training"
    # Full Image URI with unique tag
    IMAGE_URI="${BASE_IMAGE_URI}:${IMAGE_TAG}"
fi

# pipeline.py uses config("PIPELINE_ROOT"), ensure it matches or defaulted here for the runner
PIPELINE_ROOT=${PIPELINE_ROOT:-"gs://${BUCKET_NAME}/pipeline_root"}

# Export IMAGE_URI so pipeline.py picks it up (it prefers os.environ now)
export TENSORFLOW_IMAGE_URI=$IMAGE_URI

# Service Account Logic:
# If SERVICE_ACCOUNT is not set in .env, we default to empty to let Vertex AI use the default Compute Engine SA.
# We do NOT want to force a dummy "your-service-account@" string.
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-""}

echo "--------------------------------------------------------"
echo "Configuration:"
echo "Project:   $PROJECT_ID"
echo "Region:    $REGION"
echo "Image URI: $IMAGE_URI"
echo "Run Mode:  Skip Build=$SKIP_BUILD, Submit Job=$SUBMIT_JOB"
echo "--------------------------------------------------------"

# Check for local KFP and AIPlatform
if ! python3 -c "import kfp; import google.cloud.aiplatform" &> /dev/null; then
    echo "Warning: KFP or AI Platform SDK not found. Installing..."
    python3 -m pip install kfp google-cloud-aiplatform python-dotenv
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
python3 pipeline.py

echo "Pipeline compiled to 'headway_pipeline.json'."

# 3. Submit to Vertex AI (Conditional)
if [ "$SUBMIT_JOB" = true ]; then
    echo "Submitting pipeline job to Vertex AI..."

    # Submit using Python SDK (more reliable than gcloud beta)
    python3 -c "
from google.cloud import aiplatform
import sys

try:
    print(f'Submitting pipeline job to {sys.argv[1]}...')
    aiplatform.init(project='${PROJECT_ID}', location='${REGION}')
    
    # Generate Run Name dynamically
    import time
    ts = int(time.time())
    run_name_id = f'headway-run-{ts}'
    
    job = aiplatform.PipelineJob(
        display_name=f'headway-training-{ts}',
        template_path='headway_pipeline.json',
        pipeline_root='${PIPELINE_ROOT}',
        enable_caching=True,
        parameter_values={
            'project_id': '${PROJECT_ID}',
            'region': '${REGION}',
            'run_name': run_name_id
        }
    )
    sa_arg = '${SERVICE_ACCOUNT}'
    if sa_arg and sa_arg != 'None':
        print(f'Submitting with Service Account: {sa_arg}')
        job.submit(service_account=sa_arg)
    else:
        print('Submitting with Default Service Account...')
        job.submit()
except Exception as e:
    print(f'Error submitting job: {e}')
    sys.exit(1)
" "$PROJECT_ID"

    echo ""
    echo "✅ Job Submitted Successfully!"

else
    echo ""
    echo "Dry run complete. Use normal execution or remove --dry-run to submit to Vertex AI."
fi

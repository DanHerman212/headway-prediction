#!/bin/bash
set -e

# build_and_push_serving.sh
# -------------------------
# Builds the PyTorch serving image (used by the Vertex AI prediction endpoint)
# and pushes it to Artifact Registry via Cloud Build.

PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
REPO="mlops-images"
IMAGE="headway-serving"
TAG="latest"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"
DOCKERFILE="mlops_pipeline/src/serving/Dockerfile"
CONTEXT_DIR="mlops_pipeline/src/serving"

echo "=========================================================="
echo " Building Serving Image"
echo " Dockerfile: ${DOCKERFILE}"
echo " Target:     ${IMAGE_URI}"
echo "=========================================================="

# Ensure the Artifact Registry repo exists
gcloud artifacts repositories describe "${REPO}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" > /dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --repository-format=docker \
    --description="ML pipeline images"

# Build and push via Cloud Build (avoids local Docker/arch issues)
gcloud builds submit \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --tag="${IMAGE_URI}" \
    "${CONTEXT_DIR}"

echo ""
echo "Build complete. Image: ${IMAGE_URI}"

#!/bin/bash
set -e

# build_and_push_cpr.sh
# ---------------------
# Builds the CPR (Custom Prediction Routine) serving image via Cloud Build
# and pushes it to Artifact Registry.
#
# Usage:
#   bash infra/build_and_push_cpr.sh

PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
REPO="mlops-images"
IMAGE="headway-serving-cpr"
TAG="latest"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"
DOCKERFILE="mlops_pipeline/src/serving/Dockerfile.cpr"
CONTEXT_DIR="mlops_pipeline/src/serving"

echo "=========================================================="
echo " Building CPR Serving Image via Cloud Build"
echo " Dockerfile: ${DOCKERFILE}"
echo " Context:    ${CONTEXT_DIR}"
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

# Build and push via Cloud Build
gcloud builds submit \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --config=/dev/stdin \
    "${CONTEXT_DIR}" <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${IMAGE_URI}', '-f', 'Dockerfile.cpr', '.']
images:
  - '${IMAGE_URI}'
EOF

echo ""
echo "Build complete. Image: ${IMAGE_URI}"

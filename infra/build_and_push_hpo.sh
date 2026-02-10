#!/bin/bash
set -e

# build_and_push_hpo.sh
# ---------------------
# Uses Cloud Build to build the HPO Docker image and push it to Artifact Registry.
# This avoids local Docker resource issues and ensures architecture compatibility.

PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
REPO="mlops-images"
IMAGE="hpo-trial"
TAG="latest"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "=========================================================="
echo " Submitting Cloud Build for HPO Image"
echo " Target: ${IMAGE_URI}"
echo "=========================================================="

gcloud builds submit \
    --config infra/cloudbuild_hpo.yaml \
    --substitutions _IMAGE_URI="${IMAGE_URI}" \
    .

echo ""
echo "✅ Build submission complete. If successful, image is at:"
echo "   ${IMAGE_URI}"

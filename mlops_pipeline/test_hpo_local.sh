#!/bin/bash
set -e

# test_hpo_local.sh
# -----------------
# 1. Builds the HPO Docker image.
# 2. Runs a lightweight import check to ensure dependencies are installed correctly.
#    This avoids a "CrashLoop" in Vertex AI due to missing packages or import errors.

IMAGE_NAME="hpo-trial-local:test"

echo "=== 1. Building Docker Image ==="
echo "Building ${IMAGE_NAME}..."
# Use --platform linux/amd64 to match the PyTorch base image and Vertex AI target
docker build --platform linux/amd64 -f mlops_pipeline/Dockerfile.hpo -t "$IMAGE_NAME" .

echo ""
echo "=== 2. Verifying Imports & Environment ==="
# We run a simple one-liner that attempts to import the entrypoint module.
# If dependencies (ZenML, Hydra, PyTorch Forecasting) are missing/broken, this will fail.
docker run --platform linux/amd64 --rm "$IMAGE_NAME" python -c "
import sys
import logging
print('Python version:', sys.version)

print('Attempting imports...')
try:
    import hydra
    import hypertune
    import pytorch_forecasting
    import mlops_pipeline.src.hpo_entrypoint
    print('✅ SUCCESS: All critical modules imported.')
except ImportError as e:
    print(f'❌ FAILURE: Could not import module: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ FAILURE: Unexpected error during import: {e}')
    sys.exit(1)
"

echo ""
echo "✅ Build & Import Check Complete. Image is ready for registry."

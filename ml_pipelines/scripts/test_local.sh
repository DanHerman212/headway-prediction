#!/bin/bash
set -e  # Exit strictly on any error

# 0. Setup Environment & Paths
# ==============================================================================
# Determine project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in ml_pipelines/scripts, so project root is 2 levels up
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Running from Project Root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Configuration
# ==============================================================================
# Load .env file if it exists
if [ -f "${PROJECT_ROOT}/ml_pipelines/.env" ]; then
    echo "Loading configuration from ml_pipelines/.env..."
    set -a # automatically export all variables
    source "${PROJECT_ROOT}/ml_pipelines/.env"
    set +a
fi

# Fallback to defaults or environment variables
PROJECT_ID=${GCP_PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-"realtime-headway-prediction"}}
# Store artifacts in ml_pipelines/local_artifacts
ARTIFACT_DIR="${PROJECT_ROOT}/ml_pipelines/local_artifacts"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="local-test-${TIMESTAMP}"

echo "========================================================================"
echo "STARTING LOCAL PIPELINE TEST"
echo "Project: $PROJECT_ID"
echo "Artifacts: $ARTIFACT_DIR"
echo "========================================================================"

# Check for virtual environment and activate if present
if [ -d "ml_pipelines/venv" ]; then
    echo "Activating venv from ml_pipelines/venv..."
    source ml_pipelines/venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv from venv..."
    source venv/bin/activate
fi

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Prepare folders
mkdir -p "${ARTIFACT_DIR}/raw_data"
mkdir -p "${ARTIFACT_DIR}/processed_data"
mkdir -p "${ARTIFACT_DIR}/models/${RUN_NAME}"
mkdir -p "${ARTIFACT_DIR}/evaluation"
mkdir -p "${ARTIFACT_DIR}/metrics"

# 1. Extraction
# ==============================================================================
echo -e "\n[Step 1/4] Running Extraction..."
python3 -m ml_pipelines.data.data \
    --project_id "$PROJECT_ID" \
    --output_csv "${ARTIFACT_DIR}/raw_data/raw.csv"

# 2. Preprocessing
# ==============================================================================
echo -e "\n[Step 2/4] Running Preprocessing..."
python3 -m ml_pipelines.data.preprocessing \
    --input_csv "${ARTIFACT_DIR}/raw_data/raw.csv" \
    --output_csv "${ARTIFACT_DIR}/processed_data/X.csv"

# 3. Training
# ==============================================================================
echo -e "\n[Step 3/4] Running Training..."
python3 -m ml_pipelines.training.train \
    --input_csv "${ARTIFACT_DIR}/processed_data/X.csv" \
    --model_dir "${ARTIFACT_DIR}/models/${RUN_NAME}" \
    --test_dataset_path "${ARTIFACT_DIR}/processed_data/test_set.csv" \
    --epochs 2 \
    --batch_size 64

# 4. Evaluation
# ==============================================================================
echo -e "\n[Step 4/4] Running Evaluation..."
python3 -m ml_pipelines.evaluation.evaluate_model \
    --model "${ARTIFACT_DIR}/models/${RUN_NAME}" \
    --data "${ARTIFACT_DIR}/processed_data/test_set.csv" \
    --pre_split \
    --output "${ARTIFACT_DIR}/evaluation" \
    --metrics_output "${ARTIFACT_DIR}/metrics/metrics.json"

echo "========================================================================"
echo "LOCAL TEST COMPLETED SUCCESSFULLY"
echo "Check artifacts at: $ARTIFACT_DIR"
echo "========================================================================"

#!/usr/bin/env bash
# deploy_poller.sh — Provision Pub/Sub resources and a GCE VM running the GTFS poller.
#
# Prereqs: gcloud CLI authenticated, project set.
#
# Usage:
#   bash infra/deploy_poller.sh
#
# What this does:
#   1. Creates Pub/Sub topic + subscription (idempotent)
#   2. Creates a small GCE VM (e2-micro)
#   3. Copies the poller code to the VM
#   4. Installs dependencies and starts the poller as a systemd service
#
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-realtime-headway-prediction}"
REGION="us-east1"
ZONE="us-east1-b"
VM_NAME="gtfs-poller"
SA_EMAIL="mlops-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Pub/Sub resources
TOPIC_ACE="gtfs-rt-ace"
SUB_ACE="gtfs-rt-ace-sub"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
POLLER_DIR="${PROJECT_ROOT}/realtime_ingestion/poller"

echo "============================================"
echo "Deploying GTFS Poller"
echo "  Project:  ${PROJECT_ID}"
echo "  Zone:     ${ZONE}"
echo "  VM:       ${VM_NAME}"
echo "============================================"

# -----------------------------------------------------------
# 1. Pub/Sub — create topic and subscription (skip if exists)
# -----------------------------------------------------------
echo ""
echo "--- Step 1: Pub/Sub resources ---"

if gcloud pubsub topics describe "${TOPIC_ACE}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Topic '${TOPIC_ACE}' already exists."
else
    echo "Creating topic '${TOPIC_ACE}'..."
    gcloud pubsub topics create "${TOPIC_ACE}" --project="${PROJECT_ID}"
fi

if gcloud pubsub subscriptions describe "${SUB_ACE}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Subscription '${SUB_ACE}' already exists."
else
    echo "Creating subscription '${SUB_ACE}' on topic '${TOPIC_ACE}'..."
    gcloud pubsub subscriptions create "${SUB_ACE}" \
        --topic="${TOPIC_ACE}" \
        --project="${PROJECT_ID}" \
        --ack-deadline=60 \
        --message-retention-duration=1h
fi

# -----------------------------------------------------------
# 2. GCE VM — create (skip if exists)
# -----------------------------------------------------------
echo ""
echo "--- Step 2: GCE VM ---"

if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "VM '${VM_NAME}' already exists."
else
    echo "Creating VM '${VM_NAME}'..."
    gcloud compute instances create "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --machine-type=e2-micro \
        --service-account="${SA_EMAIL}" \
        --scopes=cloud-platform \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --metadata=google-logging-enabled=true \
        --tags=gtfs-poller
    echo "Waiting 30s for VM to be ready..."
    sleep 30
fi

# -----------------------------------------------------------
# 3. Copy poller code to VM
# -----------------------------------------------------------
echo ""
echo "--- Step 3: Copying poller code ---"

gcloud compute scp --recurse "${POLLER_DIR}" "${VM_NAME}:~/poller" \
    --zone="${ZONE}" --project="${PROJECT_ID}"

# -----------------------------------------------------------
# 4. Install dependencies and create systemd service
# -----------------------------------------------------------
echo ""
echo "--- Step 4: Install + systemd setup ---"

gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --command="
set -e

# Install Python + pip if needed
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv

# Create venv and install deps
cd ~/poller
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/gtfs-poller.service > /dev/null << 'UNIT'
[Unit]
Description=GTFS-RT ACE Feed Poller
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/home/$(whoami)/poller
Environment=GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
ExecStart=/home/$(whoami)/poller/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable gtfs-poller
sudo systemctl start gtfs-poller
echo 'Poller service started.'
sleep 2
sudo systemctl status gtfs-poller --no-pager || true
"

echo ""
echo "============================================"
echo "Deployment complete."
echo ""
echo "  Pub/Sub topic:        ${TOPIC_ACE}"
echo "  Pub/Sub subscription: ${SUB_ACE}"
echo "  VM:                   ${VM_NAME} (${ZONE})"
echo ""
echo "  Control:  bash infra/poller_control.sh [start|stop|status|logs]"
echo "  Pipeline: python -m pipelines.beam.streaming.streaming_pipeline \\"
echo "              --input_subscription projects/${PROJECT_ID}/subscriptions/${SUB_ACE}"
echo "============================================"

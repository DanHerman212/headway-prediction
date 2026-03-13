#!/bin/bash
# gw_poller_startup.sh -- VM startup script for Graph WaveNet grid publisher.
#
# Downloads the grid publisher code from GCS, installs deps, and starts
# a systemd service that publishes 137-row dense grid snapshots to
# Pub/Sub every minute.
#
# Logs: /var/log/gw-poller-setup.log (setup), journalctl -u gw-grid-publisher (runtime)
set -e
exec > /var/log/gw-poller-setup.log 2>&1
echo "=== Graph WaveNet grid publisher startup $(date) ==="

PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" \
    -H "Metadata-Flavor: Google")
TOPIC=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gw-topic" \
    -H "Metadata-Flavor: Google" 2>/dev/null || echo "graph-wavenet-snapshots")
BUCKET="gs://${PROJECT_ID}-gw-staging"

echo "Project: ${PROJECT_ID}"
echo "Topic:   ${TOPIC}"
echo "Bucket:  ${BUCKET}"

# Download code
mkdir -p /opt/gw-publisher
gsutil -q cp "${BUCKET}/gw-publisher.tar.gz" /tmp/gw-publisher.tar.gz
tar -xzf /tmp/gw-publisher.tar.gz -C /opt/gw-publisher
rm /tmp/gw-publisher.tar.gz

# Install Python + venv
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv

# Create venv, install deps
cd /opt/gw-publisher
python3 -m venv .venv
.venv/bin/pip install --quiet \
    google-cloud-pubsub>=2.18.0 \
    requests>=2.31.0 \
    'protobuf>=6.33.0'

# Create systemd service
cat > /etc/systemd/system/gw-grid-publisher.service <<SVC
[Unit]
Description=Graph WaveNet Grid Publisher
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/gw-publisher
ExecStart=/opt/gw-publisher/.venv/bin/python grid_publisher.py \
    --project ${PROJECT_ID} \
    --topic ${TOPIC}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable gw-grid-publisher
systemctl start gw-grid-publisher

echo "=== Graph WaveNet grid publisher startup complete $(date) ==="

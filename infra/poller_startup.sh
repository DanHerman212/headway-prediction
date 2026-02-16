#!/bin/bash
# poller_startup.sh — VM startup script for the GTFS poller.
# Runs automatically on VM creation. Logs to /var/log/poller-setup.log.
set -e
exec > /var/log/poller-setup.log 2>&1
echo "=== Poller startup script begin $(date) ==="

# Download poller code from GCS
STAGING=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
BUCKET="gs://${STAGING}-poller-staging"
mkdir -p /opt/poller
gsutil -q cp "${BUCKET}/poller-code.tar.gz" /tmp/poller-code.tar.gz
tar -xzf /tmp/poller-code.tar.gz -C /opt/poller --strip-components=1
rm /tmp/poller-code.tar.gz

# Install Python
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv

# Create venv, install deps
cd /opt/poller
python3 -m venv .venv
.venv/bin/pip install --quiet -r requirements.txt

# Create systemd service
PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
cat > /etc/systemd/system/gtfs-poller.service <<SVC
[Unit]
Description=GTFS-RT ACE Feed Poller
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/poller
Environment=GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
ExecStart=/opt/poller/.venv/bin/python run_ace.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable gtfs-poller
systemctl start gtfs-poller

echo "=== Poller startup script complete $(date) ==="

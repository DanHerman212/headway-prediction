#!/bin/bash
# Setup script for GTFS Poller VM
# This script is meant to be run after SSHing into the VM

set -e

echo "=== Setting up GTFS Poller VM ==="

# Configuration
APP_DIR="/opt/gtfs-poller"
VENV_DIR="${APP_DIR}/venv"
LOG_DIR="/var/log/gtfs-poller"

# Create directories
sudo mkdir -p "${APP_DIR}"
sudo mkdir -p "${LOG_DIR}"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    htop \
    tmux

# Create virtual environment
echo "Creating virtual environment..."
cd "${APP_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install \
    google-cloud-pubsub>=2.18.0 \
    gtfs-realtime-bindings>=1.0.0 \
    protobuf>=4.24.0 \
    requests>=2.31.0 \
    python-dateutil>=2.8.0

# Verify installation
echo "Verifying installation..."
python3 -c "from google.cloud import pubsub_v1; print('Pub/Sub OK')"
python3 -c "from google.transit import gtfs_realtime_pb2; print('GTFS-RT OK')"
python3 -c "import requests; print('Requests OK')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy the poller application to ${APP_DIR}/"
echo "2. Set environment variables in /etc/systemd/system/gtfs-poller.service"
echo "3. Start the service: sudo systemctl start gtfs-poller"
echo ""
echo "Required environment variables:"
echo "  - GOOGLE_CLOUD_PROJECT"
echo "  - GTFS_TOPIC"
echo "  - ALERTS_TOPIC"
echo "  - MTA_API_KEY"
echo ""

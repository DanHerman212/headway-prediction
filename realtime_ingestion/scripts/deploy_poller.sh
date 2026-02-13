#!/bin/bash
# Deploy poller application to the VM
# Run from local machine: ./deploy_poller.sh <VM_NAME> <ZONE>

set -e

VM_NAME="${1:-gtfs-poller-vm}"
ZONE="${2:-us-east1-b}"
APP_DIR="/opt/gtfs-poller"
LOCAL_DIR="$(dirname "$0")/../../ingestion/poller"

echo "=== Deploying Poller Application ==="
echo "VM: ${VM_NAME}"
echo "Zone: ${ZONE}"
echo "Source: ${LOCAL_DIR}"
echo ""

# Copy files to VM
echo "Copying application files..."
gcloud compute scp --zone="${ZONE}" --recurse \
    "${LOCAL_DIR}/"*.py \
    "${LOCAL_DIR}/requirements.txt" \
    "${VM_NAME}:/tmp/poller/"

# Install and configure on VM
echo "Installing on VM..."
gcloud compute ssh --zone="${ZONE}" "${VM_NAME}" --command="
    sudo mkdir -p ${APP_DIR}
    sudo cp /tmp/poller/*.py ${APP_DIR}/
    sudo cp /tmp/poller/requirements.txt ${APP_DIR}/
    
    # Activate venv and install dependencies
    source ${APP_DIR}/venv/bin/activate
    pip install -r ${APP_DIR}/requirements.txt
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    echo 'Deployment complete!'
"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To start the poller:"
echo "  gcloud compute ssh --zone=${ZONE} ${VM_NAME} --command='sudo systemctl start gtfs-poller'"
echo ""
echo "To check status:"
echo "  gcloud compute ssh --zone=${ZONE} ${VM_NAME} --command='sudo systemctl status gtfs-poller'"
echo ""
echo "To view logs:"
echo "  gcloud compute ssh --zone=${ZONE} ${VM_NAME} --command='sudo tail -f /var/log/gtfs-poller/stdout.log'"
echo ""

#!/bin/bash

# --- Configuration Variables ---
INSTANCE_NAME="my-gpu-workbench-instance"    # Choose a unique name for your instance
PROJECT_ID="realtime-headway-prediction"      # Your project ID
LOCATION="us-central1"                        # Iowa (us-central1) is the region
MACHINE_TYPE="a2-highgpu-1g"                  # 12 vCPUs, 85 GB RAM
ACCELERATOR_TYPE="NVIDIA_TESLA_A100"          # NVIDIA A100 40GB
ACCELERATOR_COUNT=1                           # Number of GPUs
DISK_TYPE="pd-balanced"                       # Balanced Persistent Disk
DISK_SIZE_GB=100                              # 100 GiB storage

# --- gcloud command to create the Vertex AI Workbench instance ---
echo "Creating Vertex AI Workbench instance: ${INSTANCE_NAME}..."

gcloud workbench instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${LOCATION}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --accelerator-count="${ACCELERATOR_COUNT}" \
  --boot-disk-type="${DISK_TYPE}" \
  --boot-disk-size="${DISK_SIZE_GB}GB" \
  --vm-image-project="cloud-notebooks-managed" \
  --vm-image-family="workbench-instances" \
  --enable-idle-shutdown \
  --idle-shutdown-timeout=360 # Shuts down after 6 hours (360 minutes) of inactivity

# --- Optional: Wait for the instance to be ready ---
echo "Instance creation initiated. It may take a few minutes for the instance to be fully provisioned."
echo "You can check its status in the Google Cloud Console or using:"
echo "gcloud workbench instances describe ${INSTANCE_NAME} --project=${PROJECT_ID} --location=${LOCATION}"

echo "Script complete."

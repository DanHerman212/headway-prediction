FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including gcloud CLI for bq command
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    gnupg \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (includes bq CLI)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ingestion scripts
COPY *.py .

# Copy SQL files for transforms
COPY sql/ /app/sql/

# Default command: run the weekly pipeline
CMD ["python", "weekly_pipeline.py"]

# Training container for headway prediction experiments
# Base: TensorFlow GPU image with Python 3.10

FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

# Set working directory
WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Note: Data is NOT copied into container
# In production, data is loaded from GCS (gs://st-convnet-training-configuration/headway-prediction/data)
# For local testing, mount data via: docker run -v $(pwd)/data:/app/data ...

# Set Python path so modules are discoverable
ENV PYTHONPATH=/app

# Set matplotlib to non-interactive backend
ENV MPLBACKEND=Agg

# Run the KFP pipeline
ENTRYPOINT ["python", "-m", "src.experiments.pipeline"]

# Use Vertex AI compatible base image with GPU support
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
# We copy the ml_pipelines directory into the container
# The layout will be /app/ml_pipelines/...
COPY . ./ml_pipelines

# Set Python path to include /app so imports modify ml_pipelines work
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default entrypoint (can be overridden by KFP command)
ENTRYPOINT ["python", "-m"]

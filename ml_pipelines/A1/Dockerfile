# Dockerfile for A1 Training Pipeline
# Builds container image with all dependencies for Vertex AI Pipelines

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install KFP SDK for pipeline components
RUN pip install --no-cache-dir kfp==2.4.0

# Copy source modules
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Default command (will be overridden by pipeline components)
CMD ["python3"]

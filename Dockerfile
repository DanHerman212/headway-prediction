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

# Copy data for local testing (optional - typically mounted via GCS)
COPY data/ ./data/

# Set Python path so modules are discoverable
ENV PYTHONPATH=/app

# Set matplotlib to non-interactive backend
ENV MPLBACKEND=Agg

# Flexible entrypoint - allows running any experiment script
# Usage: 
#   docker run <image> python -m src.experiments.run_baseline --local
#   docker run <image> python -m src.experiments.run_experiment
ENTRYPOINT ["python"]
CMD ["-m", "src.experiments.run_baseline", "--help"]

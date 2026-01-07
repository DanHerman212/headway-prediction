# Training container for headway prediction experiments
# Base: TensorFlow GPU image with Python 3.10

FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path so modules are discoverable
ENV PYTHONPATH=/app

# Default command (overridden by Vertex AI job args)
ENTRYPOINT ["python", "-m", "src.experiments.run_experiment"]

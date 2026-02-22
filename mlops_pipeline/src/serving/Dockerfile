FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps (minimal — no GPU needed for CPU inference, but base image has CUDA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — match training environment for pytorch-forecasting
# Only install what's needed for inference (no kfp, no hydra, no plotting)
# Quotes are required around specs with < or > to prevent shell redirection.
RUN pip install --no-cache-dir \
    "pytorch-forecasting==1.0.0" \
    "lightning==2.1.2" \
    "flask>=3.0" \
    "gunicorn>=22.0" \
    "pandas<2.2.0" \
    "numpy<2.0.0" \
    "google-cloud-storage>=2.9.0"

COPY predictor.py .

# Vertex AI sets AIP_STORAGE_URI and AIP_HTTP_PORT at runtime
ENV AIP_HTTP_PORT=8080

EXPOSE 8080

# gunicorn with 2 workers (model is ~200MB in memory, keep footprint reasonable)
# --preload: load model once in master, fork to workers (shared memory)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--preload", \
     "--timeout", "120", "predictor:app"]

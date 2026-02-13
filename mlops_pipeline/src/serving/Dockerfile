FROM python:3.11-slim

WORKDIR /app

# Install only what's needed for serving — no PyTorch, no Lightning, no ZenML
RUN pip install --no-cache-dir \
    onnxruntime>=1.17.0 \
    flask>=3.0 \
    gunicorn>=22.0 \
    numpy>=1.24 \
    google-cloud-storage>=2.9.0

COPY predictor.py .

# Vertex AI sets AIP_STORAGE_URI and AIP_HTTP_PORT at runtime
ENV AIP_HTTP_PORT=8080

EXPOSE 8080

# Use gunicorn for production (4 workers, preload to share model in memory)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--preload", \
     "--timeout", "120", "predictor:app"]

# For gunicorn preload to work, load the model at import time
# The predictor module calls _load_model() when imported by gunicorn --preload

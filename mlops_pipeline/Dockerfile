
FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /app

# Upgrade pip and build tools to ensure binary compatibility for cffi/cryptography
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command (overridden by pipeline)
CMD ["python", "src/train.py", "--help"]

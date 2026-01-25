
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default command (overridden by pipeline)
CMD ["python", "src/train.py", "--help"]

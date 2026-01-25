# MLOps Pipeline Refactor

This directory contains the refactored MLOps pipeline for Headway Prediction.

## Key Changes (Audit Results)

1.  **Pure Container Components**: The KFP pipeline (`pipeline.py`) now uses `@dsl.container_component`. This removes the brittle Python function wrappers and directly executes the Docker container. This ensures that what you run locally (in Docker) is exactly what runs in the pipeline.
2.  **Strict Configuration Injection**: The `Dockerfile` **no longer copies the .env file**. Instead, configuration (GCP Project, Bucket, etc.) is injected by the pipeline orchestrator (`pipeline.py`) as environment variables at runtime. This prevents secrets or stale configs from being baked into the image.
3.  **Shared Logic**: Common logic for windowing and metrics has been moved to `src/data_utils.py`. Both `src/train.py` and `src/eval.py` import from here, guaranteeing that evaluation metrics are calculated exactly the same way as training metrics.

## Structure

*   `pipeline.py`: Defines the KFP pipeline using `ContainerSpec`. Compiles to `headway_pipeline.json`.
*   `Dockerfile`: Defines the execution environment (TensorFlow + Dependencies).
*   `src/`:
    *   `extract.py`: Dummy data extraction (replace with BigQuery logic from `pipelines/`).
    *   `preprocess.py`: Cleans and scales data.
    *   `train.py`: Trains the LSTM model.
    *   `eval.py`: Evaluates the model and logs metrics to Vertex Experiments.
    *   `data_utils.py`: Shared windowing and transformation logic.
    *   `config.py`: Configuration dataclass.

## Usage

1.  **Configure**:
    Create a `.env` file in this directory with the following variables:
    ```bash
    PROJECT_ID=your-project-id
    REGION=us-central1
    BUCKET_NAME=your-bucket-name
    TENSORFLOW_IMAGE_URI=us-docker.pkg.dev/your-project/repo/headway-training:latest
    PIPELINE_ROOT=gs://your-bucket/pipeline_root
    EXPERIMENT_NAME=headway-prediction-exp
    ```

2.  **Build & Compile**:
    Run the helper script to build the container, push it to GCR/GAR, and compile the pipeline JSON.
    ```bash
    chmod +x build_run_pipeline.sh
    ./build_run_pipeline.sh
    ```

3.  **Run**:
    Upload `headway_pipeline.json` to the Vertex AI Pipelines UI.

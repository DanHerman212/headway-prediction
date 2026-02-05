# Operationalization Plan: Headway Prediction Model (TFT)

**Objective**: Migrate the successful PyTorch Forecasting (TFT) notebook into a production-grade ML pipeline using ZenML and MLflow on Google Cloud Platform (GCP).

## Phase 1: Modularization & Refactoring
*Goal: Move from "Notebook Code" to "Library Code".*

1.  **Extract Data Processing Logic**:
    *   Move the `Correct Time Index` (Physical Time) logic into a robust pre-processing function.
    *   Formalize the `Missing Value Imputation` (median filling, 23rd St logic) into a `clean_data()` function.
    *   **File**: `src/data_processing.py`

2.  **Model Definition**:
    *   Move the `TFTDisablePlotting` class into a proper module to ensure the BFloat16/Matplotlib fix is preserved in production.
    *   **File**: `src/model_definitions.py`

3.  **Training Loop**:
    *   Wrap the PyTorch Lightning `Trainer` setup into a function that accepts hyperparameters (batch_size, learning_rate) as arguments.

## Phase 2: ZenML Pipeline Construction
*Goal: Define the steps of the workflow.*

We will create a ZenML pipeline with the following steps:
1.  **`ingest_data_step`**: Loads the parquet files (or queries BigQuery).
2.  **`process_data_step`**: Applies the time index correction and imputation. Returns `TimeSeriesDataSet`.
3.  **`train_model_step`**:
    *   Initializes `TemporalFusionTransformer`.
    *   Runs `trainer.fit()`.
    *   **Crucial**: Logs metrics to MLflow automatically via generic tracking.
4.  **`evaluate_model_step`**:
    *   Loads the best checkpoint.
    *   Calculates MAE/sMAPE.
    *   Compares against the "Naive Baseline" (sanity check).

## Phase 3: MLflow & Experiment Tracking
*Goal: Ensure every run is recorded.*

1.  **ZenML Stack Configuration**:
    *   Register an MLflow Experiment Tracker in the ZenML stack.
    *   `zenml experiment-tracker register mlflow_tracker --flavor=mlflow`
2.  **Autologging**:
    *   Enable `mlflow.pytorch.autolog()` in the training step to capture loss curves and hyperparameters automatically.

## Phase 4: Google Cloud Platform (GCP) Deployment
*Goal: Run on the cloud, not on the laptop.*

1.  **Artifact Store**: Update ZenML to store dataset and model artifacts in Google Cloud Storage (GCS).
2.  **Containerization**:
    *   Ensure the `Dockerfile` in `mlops_pipeline` includes `pytorch-forecasting`, `zenml`, and `mlflow`.
3.  **Orchestration**:
    *   (Option A - Simple) Run the ZenML pipeline on a Vertex AI Workbench instance (VM).
    *   (Option B - Advanced) Deploy the ZenML pipeline to Vertex AI Pipelines for serverless execution.

## Next Actions (Tomorrow)
1.  Create the `src/` folder structure.
2.  Copy the cleaned logic from `headway_tft_training.ipynb` into Python scripts.
3.  Initialize `zenml init` in the repository.

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

## Phase 3: Configuration Management with Hydra
*Goal: Decouple configuration from code.*

To enable parameter tuning without rebuilding Docker images, we will use **Hydra** for configuration management.

1.  **Config Directory Structure**:
    Create a `conf/` directory in `mlops_pipeline`:
    ```
    mlops_pipeline/
    ├── conf/
    │   ├── config.yaml          # Main entry point
    │   ├── processing/
    │   │   └── default.yaml     # Data processing params (lookback window, missing values)
    │   ├── model/
    │   │   └── tft.yaml         # Model hyperparameters (hidden_size, attention_heads)
    │   └── training/
    │       └── default.yaml     # Trainer params (batch_size, learning_rate, max_epochs)
    ```

2.  **Hydra Integration**:
    *   Decorate the pipeline run function with `@hydra.main`.
    *   Pass the `DictConfig` object to ZenML steps.
    *   Example: `python run_pipeline.py training.batch_size=512 model.hidden_size=256`

## Phase 4: MLflow & Experiment Tracking
*Goal: Ensure every run is recorded.*

1.  **ZenML Stack Configuration**:
    *   Register an MLflow Experiment Tracker in the ZenML stack.
    *   `zenml experiment-tracker register mlflow_tracker --flavor=mlflow`
2.  **Autologging**:
    *   Enable `mlflow.pytorch.autolog()` in the training step to capture loss curves and hyperparameters automatically.

## Phase 5: HPO with Vertex AI Vizier
*Goal: Automate the search for the best model configuration using Cloud Infrastructure.*

1.  **Vertex AI Hyperparameter Tuning Job**:
    *   Instead of running the sweep inside the container, we will submit a **Hyperparameter Tuning Job** to Vertex AI.
    *   Vertex Vizier will orchestrate multiple parallel containers, each running the ZenML pipeline with different arguments overrides (e.g., `training.learning_rate=0.01`).
2.  **Hydra's Role**:
    *   Hydra remains the configuration manager. It simply accepts the command-line overrides passed by Vizier.
3.  **Selection**:
    *   Vertex AI automatically tracks the Metric Spec (e.g., `val_loss`) and reports the best trial.

## Phase 6: Deployment to Prediction Endpoint
*Goal: Serve predictions for online inference.*

1.  **Model Registry**:
    *   Register the best model from the HPO phase into the **MLflow Model Registry** or **Vertex AI Model Registry**.
2.  **Deployment Pipeline**:
    *   Create a separate pipeline (or extension of the training pipeline) that triggers only on the "promote" signal.
    *   **Step**: `deploy_to_vertex_ai_step`.
    *   **Action**: Deploys the containerized model to a **Vertex AI Endpoint** for real-time serving.
3.  **API Contract**:
    *   Define the JSON payload structure for the prediction request (e.g., `{"station_id": "...", "timestamp": "..."}`).

## Phase 7: Monitoring & Feedback Loop
*Goal: Detect drift and trigger retraining.*

1.  **Pipeline Quality Gates (Evidently AI)**:
    *   Use **Evidently** within the pipeline (`evaluate_model_step`) to validate *new* training data and model performance against the baseline. If quality drops, the pipeline fails before deployment.
2.  **Production Watchdog (Vertex AI Monitoring)**:
    *   Enable **Vertex AI Model Monitoring** on the deployed endpoint.
    *   Continuously sample prediction requests to detect "Data Drift" (input distribution shifts) and alert if retraining is needed.
3.  **Alerting**:
    *   Configure alerts (via Email/Slack) if drift exceeds a threshold.
4.  **Retraining Trigger**:
    *   Automate the triggering of the `training_pipeline` if significant drift is detected or performance dips below a specific MAE threshold.

## Phase 8: CI/CD Integration (GitHub + GCP)
*Goal: Automate the software lifecycle.*

1.  **GitHub Actions**:
    *   **CI (Continuous Integration)**: On Pull Request -> Run unit tests (`pytest`) and linting (`flake8`/`black`). Build the Docker image.
    *   **CD (Continuous Deployment)**: On Merge to Main -> Push Docker image to Google Artifact Registry (GAR). Update the Vertex AI Pipeline definition.
2.  **Version Control**:
    *   Treat the pipeline definition and configuration (`conf/`) as code.
    *   Use **DVC** (Data Version Control) or ZenML's artifact tracking to link specific Git commits to specific Data/Model versions.

## Next Actions (Tomorrow)
1.  Create the `src/` folder structure.
2.  Copy the cleaned logic from `headway_tft_training.ipynb` into Python scripts.
3.  Initialize `zenml init` in the repository.

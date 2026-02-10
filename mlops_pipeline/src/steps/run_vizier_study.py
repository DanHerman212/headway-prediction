"""
run_vizier_study.py
-------------------
ZenML step that orchestrates the HPO process on Vertex AI.

1. Serializes input datasets to GCS.
2. Loads the search space from YAML configuration.
3. Submits a Vertex AI HyperparameterTuningJob.
4. Waits for completion and returns the winning parameters.
"""

import logging
import json
import yaml
from typing import Any, Dict

import torch
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from omegaconf import DictConfig, OmegaConf
from pytorch_forecasting import TimeSeriesDataSet
from zenml import step

# Constants
# Ideally these would come from the stack, but for now we centralize them here
# or read from the generic config if available.
PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
ARTIFACT_BUCKET = "gs://mlops-artifacts-realtime-headway-prediction"
HPO_CACHE_DIR = f"{ARTIFACT_BUCKET}/hpo_cache"
STUDY_DISPLAY_NAME_PREFIX = "headway-tft-hpo"

# The header for the worker container image
# This assumes the image is already built and pushed.
# TODO: In a mature setup, we might build this image dynamically or pull from config.
TRIAL_IMAGE_URI = f"us-east1-docker.pkg.dev/{PROJECT_ID}/mlops-images/hpo-trial:latest"

logger = logging.getLogger(__name__)


def _cache_dataset(dataset: TimeSeriesDataSet, filename: str) -> str:
    """Serializes dataset to GCS and returns the path."""
    import gcsfs
    path = f"{HPO_CACHE_DIR}/{filename}"
    fs = gcsfs.GCSFileSystem()
    with fs.open(path, "wb") as f:
        torch.save(dataset, f)
    logger.info(f"Cached dataset to {path}")
    return path


def _parse_parameter_spec(search_space_conf: DictConfig) -> Dict[str, Any]:
    """
    Converts our YAML search space definition into Vertex AI ParameterSpec.
    """
    params = {}
    raw_conf = OmegaConf.to_container(search_space_conf, resolve=True)
    
    for param_name, spec in raw_conf["parameters"].items():
        p_type = spec["type"].upper()
        scale = spec.get("scale", None) # 'log', 'linear', or None
        
        if p_type == "DOUBLE":
            params[param_name] = hpt.DoubleParameterSpec(
                min=spec["min"], 
                max=spec["max"], 
                scale=scale
            )
        elif p_type == "INTEGER":
            params[param_name] = hpt.IntegerParameterSpec(
                min=spec["min"], 
                max=spec["max"], 
                scale=scale
            )
        elif p_type == "DISCRETE":
            params[param_name] = hpt.DiscreteParameterSpec(
                values=spec["values"], 
                scale=scale
            )
        elif p_type == "CATEGORICAL":
            params[param_name] = hpt.CategoricalParameterSpec(
                values=spec["values"]
            )
        else:
            logger.warning(f"Unknown parameter type {p_type} for {param_name}. Skipping.")

    return params


@step(enable_cache=False)
def run_vizier_study_step(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig,
) -> Dict[str, Any]:
    """
    Launches a Vertex AI Vizier Study.

    Parameters
    ----------
    training_dataset : TimeSeriesDataSet
        The processed training data.
    validation_dataset : TimeSeriesDataSet
        The processed validation data.
    config : DictConfig
         The full Hydra configuration, containing 'hpo_search_space'.

    Returns
    -------
    best_params : Dict
        The hyperparameters of the best performing trial.
    """
    # Extract search space config
    if "hpo_search_space" not in config:
        raise ValueError("Config is missing 'hpo_search_space'. Ensure you are loading the correct config or overrides.")
    
    search_space_config = config.hpo_search_space

    # 1. Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=ARTIFACT_BUCKET)

    # 2. Serialize Data to GCS
    #    The trials run in isolated containers, so they need to download the data.
    train_path = _cache_dataset(training_dataset, "train.pt")
    val_path = _cache_dataset(validation_dataset, "val.pt")

    # 3. Define the Trial Specification (CustomJob)
    #    This tells Vertex what command to run for each trial.
    #    The arguments match what src/hpo_entrypoint.py expects.
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "a2-highgpu-1g",  # A100 for speed
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": TRIAL_IMAGE_URI,
            "args": [
                # Entrypoint script path inside the container
                # We use python -m to ensure imports work
                "python", "-m", "mlops_pipeline.src.hpo_entrypoint",
                "--train_dataset_path", train_path,
                "--val_dataset_path", val_path,
                # Vizier will append --param_name=value flags automatically
            ],
        },
    }]

    custom_job = aiplatform.CustomJob(
        display_name=f"{STUDY_DISPLAY_NAME_PREFIX}-trial-job",
        worker_pool_specs=worker_pool_specs,
    )

    # 4. Create the HyperparameterTuningJob
    #    Map the YAML config to the SDK objects
    parameter_spec = _parse_parameter_spec(search_space_config)
    
    # Extract metric info
    metric_id = search_space_config.metric.id
    metric_goal = search_space_config.metric.goal # 'minimize' or 'maximize'
    metric_spec = {metric_id: metric_goal}

    hpo_job = aiplatform.HyperparameterTuningJob(
        display_name=STUDY_DISPLAY_NAME_PREFIX,
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=20,  # Cap cost
        parallel_trial_count=5,
        search_algorithm=None, # Vizier default (Bayesian Optimization)
    )

    logger.info("Submitting Vizier Study...")
    hpo_job.run(sync=True) # Block until done

    # 5. Extract Results
    logger.info("Study completed. Fetching best trial...")
    best_trial = hpo_job.trials[0] # Just initialization
    
    # Sort trials by metric to find the 'best' one
    # Note: Vertex SDK doesn't always strictly order them in .trials
    trials = hpo_job.trials
    # Filter out failed/cancelled
    valid_trials = [t for t in trials if t.state.name == "SUCCEEDED"]
    
    if not valid_trials:
        raise RuntimeError("No trials succeeded!")

    sorted_trials = sorted(
        valid_trials,
        key=lambda t: t.final_measurement.metrics[0].value,
        reverse=(metric_goal == "maximize")
    )
    best_trial = sorted_trials[0]

    logger.info(f"Best Trial ID: {best_trial.id}")
    logger.info(f"Best Metric ({metric_id}): {best_trial.final_measurement.metrics[0].value}")

    # Convert parameter objects to a clean dict
    best_params = {p.parameter_id: p.value for p in best_trial.parameters}
    
    # Type correction: everything comes back as float/string, need to cast ints
    # We can infer from the search space config
    raw_conf = OmegaConf.to_container(search_space_config, resolve=True)
    for k, v in best_params.items():
        spec = raw_conf["parameters"].get(k)
        if spec and spec["type"] in ["INTEGER", "DISCRETE"]:
             # Heuristic: if defined values are ints, cast result to int
             values = spec.get("values", [])
             if values and isinstance(values[0], int):
                 best_params[k] = int(v)
             elif spec["type"] == "INTEGER":
                 best_params[k] = int(v)

    logger.info(f"Best Parameters: {best_params}")
    return best_params

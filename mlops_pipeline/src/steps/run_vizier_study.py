"""
run_vizier_study.py
-------------------
ZenML step that orchestrates Vertex AI Hyperparameter Tuning.

Flow:
    1. Serialize train/val datasets to GCS  (trials run in isolated containers)
    2. Build Vertex AI ParameterSpec from the YAML search space
    3. Submit a HyperparameterTuningJob and block until completion
    4. Return the best trial's parameters as a plain dict

All GCP project / region / image constants come from the Hydra ``infra``
config group — nothing is hardcoded here.
"""

import logging
from typing import Any, Dict

import torch
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from omegaconf import DictConfig, OmegaConf
from pytorch_forecasting import TimeSeriesDataSet
from zenml import step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_dataset(dataset: TimeSeriesDataSet, filename: str, cache_dir: str) -> str:
    """Serialize a dataset to GCS and return the gs:// path."""
    import gcsfs

    path = f"{cache_dir}/{filename}"
    fs = gcsfs.GCSFileSystem()
    with fs.open(path, "wb") as f:
        torch.save(dataset, f)
    logger.info("Cached dataset → %s", path)
    return path


def _build_parameter_spec(search_space: DictConfig) -> Dict[str, Any]:
    """Convert our YAML search-space definition into Vertex AI ParameterSpec objects."""
    params: Dict[str, Any] = {}
    raw = OmegaConf.to_container(search_space, resolve=True)

    for name, spec in raw["parameters"].items():
        kind = spec["type"].upper()
        scale = spec.get("scale", None)

        if kind == "DOUBLE":
            params[name] = hpt.DoubleParameterSpec(min=spec["min"], max=spec["max"], scale=scale)
        elif kind == "INTEGER":
            params[name] = hpt.IntegerParameterSpec(min=spec["min"], max=spec["max"], scale=scale)
        elif kind == "DISCRETE":
            params[name] = hpt.DiscreteParameterSpec(values=spec["values"], scale=scale)
        elif kind == "CATEGORICAL":
            params[name] = hpt.CategoricalParameterSpec(values=spec["values"])
        else:
            logger.warning("Unknown parameter type %s for %s — skipping", kind, name)

    return params


def _cast_best_params(best_params: Dict[str, Any], search_space: DictConfig) -> Dict[str, Any]:
    """Cast Vizier result values to the correct Python types based on the search-space spec."""
    raw = OmegaConf.to_container(search_space, resolve=True)
    for key, value in best_params.items():
        spec = raw["parameters"].get(key)
        if spec is None:
            continue
        if spec["type"].upper() == "INTEGER":
            best_params[key] = int(value)
        elif spec["type"].upper() == "DISCRETE":
            values = spec.get("values", [])
            if values and isinstance(values[0], int):
                best_params[key] = int(value)
    return best_params


# ---------------------------------------------------------------------------
# ZenML Step
# ---------------------------------------------------------------------------

@step(enable_cache=False)
def run_vizier_study_step(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig,
) -> Dict[str, Any]:
    """
    Launch a Vertex AI Vizier study and return the best hyperparameters.

    Parameters
    ----------
    training_dataset, validation_dataset : TimeSeriesDataSet
        Processed data splits.
    config : DictConfig
        Full Hydra config.  Must contain ``hpo_search_space`` and ``infra``.

    Returns
    -------
    dict  — best hyperparameters from the winning trial.
    """
    # ---- Validate required config sections ----------------------------------
    if "hpo_search_space" not in config:
        raise ValueError("Config missing 'hpo_search_space'.  "
                         "Did you forget the override '+hpo_search_space=vizier_v1'?")
    if "infra" not in config:
        raise ValueError("Config missing 'infra'.  Add 'infra: default' to conf/config.yaml defaults.")

    infra = config.infra
    search_space = config.hpo_search_space

    # ---- 1. Init Vertex AI --------------------------------------------------
    aiplatform.init(
        project=infra.project_id,
        location=infra.location,
        staging_bucket=infra.artifact_bucket,
    )

    # ---- 2. Cache datasets to GCS ------------------------------------------
    train_path = _cache_dataset(training_dataset, "train.pt", infra.hpo_cache_dir)
    val_path = _cache_dataset(validation_dataset, "val.pt", infra.hpo_cache_dir)

    # ---- 3. Define trial worker spec ----------------------------------------
    #
    #   The Dockerfile sets:
    #       ENTRYPOINT ["python", "-m", "mlops_pipeline.src.hpo_entrypoint"]
    #
    #   So `container_spec.args` should contain ONLY the arguments
    #   passed to that entrypoint.  Vizier will append --param=value
    #   flags after these.
    #
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": infra.machine_type,
                "accelerator_type": infra.accelerator_type,
                "accelerator_count": infra.accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": infra.trial_image_uri,
                "args": [
                    "--train_dataset_path", train_path,
                    "--val_dataset_path", val_path,
                    # Vizier will append --<param>=<value> after these
                ],
            },
        }
    ]

    custom_job = aiplatform.CustomJob(
        display_name=f"{infra.study_display_name}-trial-job",
        worker_pool_specs=worker_pool_specs,
    )

    # ---- 4. Submit HPO job --------------------------------------------------
    parameter_spec = _build_parameter_spec(search_space)
    metric_spec = {search_space.metric.id: search_space.metric.goal}

    hpo_job = aiplatform.HyperparameterTuningJob(
        display_name=infra.study_display_name,
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=infra.max_trial_count,
        parallel_trial_count=infra.parallel_trial_count,
        search_algorithm=None,  # Vizier default (Bayesian)
    )

    logger.info("Submitting Vizier study …")
    hpo_job.run(sync=True)

    # ---- 5. Extract best trial ----------------------------------------------
    logger.info("Study completed.  Extracting best trial …")
    valid_trials = [t for t in hpo_job.trials if t.state.name == "SUCCEEDED"]
    if not valid_trials:
        raise RuntimeError("All trials failed — no successful results.")

    metric_goal = search_space.metric.goal
    sorted_trials = sorted(
        valid_trials,
        key=lambda t: t.final_measurement.metrics[0].value,
        reverse=(metric_goal == "maximize"),
    )
    best = sorted_trials[0]

    logger.info("Best trial %s — %s = %f",
                best.id, search_space.metric.id,
                best.final_measurement.metrics[0].value)

    best_params = {p.parameter_id: p.value for p in best.parameters}
    best_params = _cast_best_params(best_params, search_space)

    logger.info("Best parameters: %s", best_params)
    return best_params

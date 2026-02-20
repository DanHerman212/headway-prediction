"""
hpo_pipeline.py
---------------
Vertex AI Pipelines (KFP v2) pipeline for Hyperparameter Optimization.

Flow:
1. Load Config (HPO profile + search space)
2. Ingest Data
3. Process Data
4. Run Vizier Study (returns best params)

Reuses the same Docker image and data processing logic as the training pipeline.
"""

import json
from typing import List, Optional

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

from .pipeline import TRAINING_IMAGE, ingest_data_op, process_data_op


@dsl.component(base_image=TRAINING_IMAGE)
def load_hpo_config_op(
    user_overrides_json: str,
    config_out: Output[Artifact],
):
    """Load Hydra config with HPO profile and search space baked in."""
    import json
    from omegaconf import OmegaConf
    from mlops_pipeline.src.steps.config_loader import load_config

    base_overrides = ["training=hpo", "+hpo_search_space=vizier_v1"]
    user_overrides = json.loads(user_overrides_json) if user_overrides_json and user_overrides_json != "null" else []
    all_overrides = base_overrides + user_overrides

    config = load_config(overrides=all_overrides)
    with open(config_out.path, "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))


@dsl.component(base_image=TRAINING_IMAGE)
def run_vizier_study_op(
    train_ds_in: Input[Dataset],
    val_ds_in: Input[Dataset],
    config_yaml: Input[Artifact],
    best_params_out: Output[Artifact],
):
    """Launch a Vertex AI Vizier study and return the best hyperparameters."""
    import json
    import torch
    from omegaconf import OmegaConf
    from mlops_pipeline.src.steps.run_vizier_study import run_vizier_study

    training_dataset = torch.load(train_ds_in.path, weights_only=False)
    validation_dataset = torch.load(val_ds_in.path, weights_only=False)
    with open(config_yaml.path) as f:
        config = OmegaConf.create(f.read())

    best_params = run_vizier_study(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=config,
    )
    with open(best_params_out.path, "w") as f:
        json.dump(best_params, f, indent=2)


@dsl.pipeline(
    name="headway-hpo-pipeline",
    description="Vizier HPO study: ingest -> process -> run Vizier -> best params",
)
def headway_hpo_pipeline(
    data_path: str,
    hydra_overrides_json: str = "null",
):
    """
    Pipeline that runs a Vizier HPO study.

    Parameters
    ----------
    data_path : str
        GCS path to the training parquet file.
    hydra_overrides_json : str
        JSON-encoded list of additional Hydra overrides.
    """
    # 1. Load config (HPO profile + search space merged inside the component)
    config_task = load_hpo_config_op(user_overrides_json=hydra_overrides_json)

    # 2. Ingest data
    ingest_task = ingest_data_op(file_path=data_path)

    # 3. Process data
    process_task = process_data_op(
        raw_data_in=ingest_task.outputs["raw_data_out"],
        config_yaml=config_task.outputs["config_out"],
    )

    # 4. Run Vizier study
    run_vizier_study_op(
        train_ds_in=process_task.outputs["train_ds_out"],
        val_ds_in=process_task.outputs["val_ds_out"],
        config_yaml=config_task.outputs["config_out"],
    )

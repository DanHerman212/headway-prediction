"""
hpo_pipeline.py
---------------
Vertex AI Pipelines (KFP v2) pipeline for Hyperparameter Optimization.

Flow:
1. Load Config (HPO profile + search space)
2. Ingest Data
3. Process Data
4. Run Vizier Study (returns best params)

Reuses the same Docker image and data processing logic as the training pipeline.Source code is installed at runtime from a GCS-hosted wheel (no rebuild needed)."""

import json
from typing import List, Optional

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

from .pipeline import TRAINING_IMAGE, ingest_data_op, process_data_op


@dsl.component(base_image=TRAINING_IMAGE)
def load_hpo_config_op(
    config_gcs_uri: str,
    config_out: Output[Artifact],
):
    """Download the fully-resolved HPO config YAML from GCS.

    run.py resolves the config locally (with HPO overrides applied)
    and uploads the result to GCS.
    """
    from google.cloud import storage as gcs_storage

    path = config_gcs_uri.replace("gs://", "")
    bucket_name = path.split("/")[0]
    blob_path = "/".join(path.split("/")[1:])

    client = gcs_storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    config_yaml = blob.download_as_text()

    with open(config_out.path, "w") as f:
        f.write(config_yaml)


@dsl.component(base_image=TRAINING_IMAGE)
def run_vizier_study_op(
    train_ds_in: Input[Dataset],
    val_ds_in: Input[Dataset],
    config_yaml: Input[Artifact],
    source_whl_uri: str,
    best_params_out: Output[Artifact],
):
    """Launch a Vertex AI Vizier study and return the best hyperparameters."""
    import subprocess, sys, tempfile, os
    from google.cloud import storage as _gcs
    _p = source_whl_uri.replace("gs://", ""); _b, _k = _p.split("/", 1)
    _f = os.path.join(tempfile.gettempdir(), os.path.basename(_k))
    _gcs.Client().bucket(_b).blob(_k).download_to_filename(_f)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _f])

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
    config_gcs_uri: str,
    source_whl_uri: str,
):
    """
    Pipeline that runs a Vizier HPO study.

    Parameters
    ----------
    data_path : str
        GCS path to the training parquet file.
    config_gcs_uri : str
        GCS URI of the fully-resolved HPO config YAML (uploaded by run.py).
    source_whl_uri : str
        GCS URI of the mlops_pipeline wheel (built and uploaded by run.py).
    """
    # 1. Load config (resolved locally by run.py with HPO overrides)
    config_task = load_hpo_config_op(config_gcs_uri=config_gcs_uri)

    # 2. Ingest data
    ingest_task = ingest_data_op(
        file_path=data_path,
        source_whl_uri=source_whl_uri,
    )

    # 3. Process data
    process_task = process_data_op(
        raw_data_in=ingest_task.outputs["raw_data_out"],
        config_yaml=config_task.outputs["config_out"],
        source_whl_uri=source_whl_uri,
    )

    # 4. Run Vizier study
    run_vizier_study_op(
        train_ds_in=process_task.outputs["train_ds_out"],
        val_ds_in=process_task.outputs["val_ds_out"],
        config_yaml=config_task.outputs["config_out"],
        source_whl_uri=source_whl_uri,
    )

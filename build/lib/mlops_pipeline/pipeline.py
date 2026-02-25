"""
pipeline.py
-----------
Vertex AI Pipelines (KFP v2) training pipeline for Headway Prediction.

Each component runs inside a pre-built Docker image that contains all
*dependencies*.  Source code is delivered at runtime via a wheel that
run.py builds and uploads to GCS — no image rebuild for code changes.
Complex objects (DataFrames, TimeSeriesDataSets, models) are serialized
to GCS-backed KFP artifacts; primitives and config pass as JSON strings.
"""

from typing import List, Optional

from kfp import dsl
from kfp.dsl import HTML, Input, Output, Dataset, Model, Artifact

# ---------------------------------------------------------------------------
# Container image — built from infra/Dockerfile.training
# ---------------------------------------------------------------------------
TRAINING_IMAGE = (
    "us-east1-docker.pkg.dev/realtime-headway-prediction/"
    "mlops-images/headway-training:latest"
)

# GPU machine spec for training step
GPU_MACHINE_TYPE = "a2-highgpu-1g"
GPU_ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"
GPU_ACCELERATOR_COUNT = 1


# ═══════════════════════════════════════════════════════════════════════════
# Components
# ═══════════════════════════════════════════════════════════════════════════

@dsl.component(base_image=TRAINING_IMAGE)
def load_config_op(
    config_gcs_uri: str,
    config_out: Output[Artifact],
):
    """Download the fully-resolved config YAML from GCS.

    run.py resolves the Hydra config locally (reading workspace YAML files
    and applying CLI overrides), then uploads the result to GCS.  This
    component simply downloads that resolved config, so the pipeline
    always uses the caller's local config — no Docker rebuild required
    for config-only changes.
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
def fetch_vizier_params_op(
    config_yaml: Input[Artifact],
    source_whl_uri: str,
    params_out: Output[Artifact],
):
    """Fetch best hyperparameters from the latest Vizier study."""
    import subprocess, sys, tempfile, os
    from google.cloud import storage as _gcs
    _p = source_whl_uri.replace("gs://", ""); _b, _k = _p.split("/", 1)
    _f = os.path.join(tempfile.gettempdir(), os.path.basename(_k))
    _gcs.Client().bucket(_b).blob(_k).download_to_filename(_f)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _f])

    import json
    from omegaconf import OmegaConf
    from mlops_pipeline.src.steps.fetch_best_vizier_params import fetch_best_vizier_params

    with open(config_yaml.path) as f:
        config = OmegaConf.create(f.read())
    best_params = fetch_best_vizier_params(config=config)
    with open(params_out.path, "w") as f:
        json.dump(best_params, f)


@dsl.component(base_image=TRAINING_IMAGE)
def ingest_data_op(
    file_path: str,
    source_whl_uri: str,
    raw_data_out: Output[Dataset],
):
    """Ingest raw parquet data and save as artifact."""
    import subprocess, sys, tempfile, os
    from google.cloud import storage as _gcs
    _p = source_whl_uri.replace("gs://", ""); _b, _k = _p.split("/", 1)
    _f = os.path.join(tempfile.gettempdir(), os.path.basename(_k))
    _gcs.Client().bucket(_b).blob(_k).download_to_filename(_f)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _f])

    from mlops_pipeline.src.steps.ingest_data import ingest_data
    df = ingest_data(file_path=file_path)
    df.to_parquet(raw_data_out.path, index=False)


@dsl.component(base_image=TRAINING_IMAGE)
def process_data_op(
    raw_data_in: Input[Dataset],
    config_yaml: Input[Artifact],
    source_whl_uri: str,
    train_ds_out: Output[Dataset],
    val_ds_out: Output[Dataset],
    test_ds_out: Output[Dataset],
    time_lookup_out: Output[Dataset],
):
    """Clean data and create train/val/test TimeSeriesDataSet splits."""
    import subprocess, sys, tempfile, os
    from google.cloud import storage as _gcs
    _p = source_whl_uri.replace("gs://", ""); _b, _k = _p.split("/", 1)
    _f = os.path.join(tempfile.gettempdir(), os.path.basename(_k))
    _gcs.Client().bucket(_b).blob(_k).download_to_filename(_f)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _f])

    import torch
    import pandas as pd
    from omegaconf import OmegaConf
    from mlops_pipeline.src.steps.process_data import process_data

    df = pd.read_parquet(raw_data_in.path)
    with open(config_yaml.path) as f:
        config = OmegaConf.create(f.read())

    train_ds, val_ds, test_ds, time_lookup = process_data(raw_data=df, config=config)

    torch.save(train_ds, train_ds_out.path)
    torch.save(val_ds, val_ds_out.path)
    torch.save(test_ds, test_ds_out.path)
    time_lookup.to_parquet(time_lookup_out.path, index=False)


@dsl.component(base_image=TRAINING_IMAGE)
def train_model_op(
    train_ds_in: Input[Dataset],
    val_ds_in: Input[Dataset],
    config_yaml: Input[Artifact],
    run_name: str,
    vizier_params_json: str,
    source_whl_uri: str,
    model_out: Output[Model],
):
    """Train the TFT model with Vertex AI experiment tracking."""
    import subprocess, sys, tempfile, os
    from google.cloud import storage as _gcs
    _p = source_whl_uri.replace("gs://", ""); _b, _k = _p.split("/", 1)
    _f = os.path.join(tempfile.gettempdir(), os.path.basename(_k))
    _gcs.Client().bucket(_b).blob(_k).download_to_filename(_f)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _f])

    import json
    import torch
    from omegaconf import OmegaConf
    from mlops_pipeline.src.steps.train_model import train_model

    training_dataset = torch.load(train_ds_in.path, weights_only=False)
    validation_dataset = torch.load(val_ds_in.path, weights_only=False)
    with open(config_yaml.path) as f:
        config = OmegaConf.create(f.read())

    vizier_params = json.loads(vizier_params_json) if vizier_params_json else None

    model = train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=config,
        run_name=run_name,
        vizier_params=vizier_params,
    )
    torch.save(model.state_dict(), model_out.path)


@dsl.component(base_image=TRAINING_IMAGE)
def evaluate_model_op(
    model_state_in: Input[Model],
    train_ds_in: Input[Dataset],
    test_ds_in: Input[Dataset],
    config_yaml: Input[Artifact],
    run_name: str,
    source_whl_uri: str,
    time_lookup_in: Input[Dataset],
    test_mae_out: Output[Artifact],
    test_smape_out: Output[Artifact],
    rush_hour_html: Output[HTML],
    interpretation_html: Output[HTML],
):
    """Evaluate on test set, produce plots, log metrics."""
    import subprocess, sys, tempfile, os
    from google.cloud import storage as _gcs
    _p = source_whl_uri.replace("gs://", ""); _b, _k = _p.split("/", 1)
    _f = os.path.join(tempfile.gettempdir(), os.path.basename(_k))
    _gcs.Client().bucket(_b).blob(_k).download_to_filename(_f)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _f])

    import json
    import torch
    import pandas as pd
    from omegaconf import OmegaConf
    from mlops_pipeline.src.model_definitions import create_model
    from mlops_pipeline.src.steps.evaluate_model import evaluate_model

    training_dataset = torch.load(train_ds_in.path, weights_only=False)
    test_dataset = torch.load(test_ds_in.path, weights_only=False)
    with open(config_yaml.path) as f:
        config = OmegaConf.create(f.read())
    time_lookup = pd.read_parquet(time_lookup_in.path)

    # Reconstruct model from state_dict using training dataset params
    model = create_model(training_dataset, config)
    model.load_state_dict(torch.load(model_state_in.path, weights_only=False))

    mae, smape, rush_html, interp_html = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        config=config,
        run_name=run_name,
        time_lookup=time_lookup,
    )

    # Write metric outputs as JSON strings
    with open(test_mae_out.path, "w") as f:
        json.dump(mae, f)
    with open(test_smape_out.path, "w") as f:
        json.dump(smape, f)

    # Write HTML artifacts
    with open(rush_hour_html.path, "w") as f:
        f.write(rush_html)
    with open(interpretation_html.path, "w") as f:
        f.write(interp_html)


@dsl.component(base_image=TRAINING_IMAGE)
def register_model_op(
    model_state_in: Input[Model],
    train_ds_in: Input[Dataset],
    config_yaml: Input[Artifact],
    test_mae_in: Input[Artifact],
    test_smape_in: Input[Artifact],
    run_id: str,
    source_whl_uri: str,
    model_resource_name: Output[Artifact],
):
    """Export to ONNX, upload to GCS, register in Vertex AI Model Registry."""
    import subprocess, sys, tempfile, os
    from google.cloud import storage as _gcs
    _p = source_whl_uri.replace("gs://", ""); _b, _k = _p.split("/", 1)
    _f = os.path.join(tempfile.gettempdir(), os.path.basename(_k))
    _gcs.Client().bucket(_b).blob(_k).download_to_filename(_f)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _f])

    import json
    import torch
    from omegaconf import OmegaConf
    from mlops_pipeline.src.model_definitions import create_model
    from mlops_pipeline.src.steps.deploy_model import register_model

    training_dataset = torch.load(train_ds_in.path, weights_only=False)
    with open(config_yaml.path) as f:
        config = OmegaConf.create(f.read())

    # Read metrics from artifact files
    with open(test_mae_in.path) as f:
        test_mae = float(json.load(f))
    with open(test_smape_in.path) as f:
        test_smape = float(json.load(f))

    model = create_model(training_dataset, config)
    model.load_state_dict(torch.load(model_state_in.path, weights_only=False))

    resource_name = register_model(
        model=model,
        training_dataset=training_dataset,
        config=config,
        test_mae=test_mae,
        test_smape=test_smape,
        run_id=run_id,
    )

    with open(model_resource_name.path, "w") as f:
        f.write(resource_name)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline DAG
# ═══════════════════════════════════════════════════════════════════════════

@dsl.pipeline(
    name="headway-training-pipeline",
    description="End-to-end TFT training: ingest -> process -> train (GPU) -> evaluate -> register",
)
def headway_training_pipeline(
    data_path: str,
    run_name: str,
    config_gcs_uri: str,
    source_whl_uri: str,
    use_vizier_params: bool = False,
):
    """
    End-to-end training pipeline for Headway Prediction.

    Parameters
    ----------
    data_path : str
        GCS path to the training parquet file.
    run_name : str
        Unique run identifier (experiment run name, artifact prefix, etc.).
    config_gcs_uri : str
        GCS URI of the fully-resolved config YAML (uploaded by run.py).
    source_whl_uri : str
        GCS URI of the mlops_pipeline wheel (built and uploaded by run.py).
    use_vizier_params : bool
        If True, fetch best params from the latest Vizier study.
    """
    # 1. Load config
    config_task = load_config_op(config_gcs_uri=config_gcs_uri)

    # 1b. Optionally fetch Vizier best params
    vizier_json = ""
    with dsl.If(use_vizier_params == True):
        vizier_task = fetch_vizier_params_op(
            config_yaml=config_task.outputs["config_out"],
            source_whl_uri=source_whl_uri,
        )

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

    # 4. Train model (GPU)
    train_task = train_model_op(
        train_ds_in=process_task.outputs["train_ds_out"],
        val_ds_in=process_task.outputs["val_ds_out"],
        config_yaml=config_task.outputs["config_out"],
        run_name=run_name,
        vizier_params_json=vizier_json,
        source_whl_uri=source_whl_uri,
    ).set_accelerator_type(GPU_ACCELERATOR_TYPE)\
     .set_accelerator_limit(GPU_ACCELERATOR_COUNT)\
     .set_cpu_limit("4")\
     .set_memory_limit("16G")

    # 5. Evaluate model
    eval_task = evaluate_model_op(
        model_state_in=train_task.outputs["model_out"],
        train_ds_in=process_task.outputs["train_ds_out"],
        test_ds_in=process_task.outputs["test_ds_out"],
        config_yaml=config_task.outputs["config_out"],
        run_name=run_name,
        source_whl_uri=source_whl_uri,
        time_lookup_in=process_task.outputs["time_lookup_out"],
    )

    # 6. Register model in Vertex AI Model Registry
    register_model_op(
        model_state_in=train_task.outputs["model_out"],
        train_ds_in=process_task.outputs["train_ds_out"],
        config_yaml=config_task.outputs["config_out"],
        test_mae_in=eval_task.outputs["test_mae_out"],
        test_smape_in=eval_task.outputs["test_smape_out"],
        run_id=run_name,
        source_whl_uri=source_whl_uri,
    )
"""train_model.py — Training step with Vertex AI experiment tracking + TensorBoard."""

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform, storage
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from ..training_core import train_tft

logger = logging.getLogger(__name__)


class VertexAIMetricsCallback(pl.Callback):
    """Streams training metrics to Vertex AI Experiments + managed TensorBoard.

    Uses aiplatform.log_time_series_metrics() which writes to both the
    Experiments run and the linked TensorBoard instance in near-real-time.
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {
            k: float(v)
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, (int, float)) or (hasattr(v, "item") and v.numel() == 1)
        }
        if metrics:
            try:
                aiplatform.log_time_series_metrics(metrics, step=trainer.current_epoch)
            except Exception as e:
                logger.debug("Failed to stream metrics to Vertex AI: %s", e)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif hasattr(v, "item") and hasattr(v, "numel") and v.numel() == 1:
                metrics[k] = float(v.item())
        if metrics:
            try:
                aiplatform.log_time_series_metrics(metrics, step=trainer.current_epoch)
            except Exception as e:
                logger.debug("Failed to stream metrics to Vertex AI: %s", e)


def _upload_dir_to_gcs(local_dir: str, gcs_uri: str) -> None:
    """Upload a local directory tree to a GCS URI (gs://bucket/prefix)."""
    if not gcs_uri.startswith("gs://"):
        logger.warning("GCS URI %s does not start with gs://, skipping upload.", gcs_uri)
        return
    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{prefix}/{rel_path}" if prefix else rel_path
            bucket.blob(blob_path).upload_from_filename(local_path)
    logger.info("Uploaded TensorBoard logs from %s to %s", local_dir, gcs_uri)


def train_model(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig,
    run_name: str,
    vizier_params: Optional[Dict[str, Any]] = None,
) -> TemporalFusionTransformer:
    """
    Train the TFT model with Vertex AI Experiments tracking.

    Parameters
    ----------
    training_dataset, validation_dataset : TimeSeriesDataSet
        Processed data splits.
    config : DictConfig
        Full Hydra config.
    run_name : str
        Unique run identifier (used for experiment run naming + GCS paths).
    vizier_params : dict, optional
        Best hyperparameters from Vizier to override config.
    """
    # 0. Apply Vizier overrides (if provided) directly onto the loaded config
    if vizier_params:
        logger.info("Applying Vizier best params to config: %s", vizier_params)
        for key, value in vizier_params.items():
            try:
                OmegaConf.update(config, key, value)
            except Exception as e:
                logger.warning("Could not apply Vizier param %s=%s: %s", key, value, e)
        logger.info("Final config after Vizier overrides:\n%s", OmegaConf.to_yaml(config))

    # 1. Initialize Vertex AI Experiments
    experiment_name = config.get("experiment_name", "headway-tft").lower().replace("_", "-")
    tb_resource = config.infra.tensorboard_resource_name

    aiplatform.init(
        project=config.infra.project_id,
        location=config.infra.location,
        experiment=experiment_name,
        experiment_tensorboard=tb_resource,
    )
    aiplatform.start_run(run_name)

    # 2. TensorBoard logger — write locally for speed, upload to GCS after training
    local_tb_dir = tempfile.mkdtemp(prefix="tb_logs_")
    tb_logger = TensorBoardLogger(
        save_dir=local_tb_dir,
        name=experiment_name,
        default_hp_metric=False,
        log_graph=False,
    )
    logger.info("TensorBoard logging locally to: %s (will upload to GCS after training)", local_tb_dir)

    # 3. Log hyperparameters to Vertex AI Experiments
    hparams = {
        "batch_size": config.training.batch_size,
        "max_epochs": config.training.max_epochs,
        "learning_rate": config.model.learning_rate,
        "gradient_clip_val": config.training.gradient_clip_val,
        "precision": config.training.precision,
        "hidden_size": config.model.hidden_size,
        "attention_head_size": config.model.attention_head_size,
        "dropout": config.model.dropout,
        "hidden_continuous_size": config.model.hidden_continuous_size,
        "lstm_layers": config.model.lstm_layers,
        "optimizer": config.model.optimizer,
        "max_encoder_length": config.processing.max_encoder_length,
    }
    feature_params = {
        "known_reals": ",".join(config.processing.time_varying_known_reals),
        "unknown_reals": ",".join(config.processing.time_varying_unknown_reals),
        "known_categoricals": ",".join(config.processing.time_varying_known_categoricals),
        "unknown_categoricals": ",".join(config.processing.time_varying_unknown_categoricals),
        "static_categoricals": ",".join(config.processing.static_categoricals),
    }
    hparams.update(feature_params)
    aiplatform.log_params(hparams)
    tb_logger.log_hyperparams(hparams)

    # 4. Train
    vertex_cb = VertexAIMetricsCallback()
    result = train_tft(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=config,
        lightning_logger=tb_logger,
        extra_callbacks=[vertex_cb],
    )

    # 5. Log final metric to Vertex AI Experiments
    aiplatform.log_metrics({"best_val_loss": result.best_val_loss})
    aiplatform.end_run()

    # 6. Upload TensorBoard logs to GCS for persistent storage
    gcs_tb_dir = config.training.tensorboard_log_dir
    try:
        _upload_dir_to_gcs(local_tb_dir, f"{gcs_tb_dir}/{experiment_name}")
    except Exception as e:
        logger.warning("Failed to upload TensorBoard logs to GCS: %s", e)
    finally:
        shutil.rmtree(local_tb_dir, ignore_errors=True)

    return result.model
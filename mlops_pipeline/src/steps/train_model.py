"""train_model.py — ZenML training step with Vertex AI experiment tracking + TensorBoard."""

import logging

from google.cloud import aiplatform
from lightning.pytorch.loggers import TensorBoardLogger
from zenml import step
from omegaconf import DictConfig
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from ..training_core import train_tft

logger = logging.getLogger(__name__)


@step(experiment_tracker="vertex_tracker", enable_cache=False)
def train_model_step(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig,
) -> TemporalFusionTransformer:
    """
    Train the TFT model.

    Vertex AI experiment tracker (managed by ZenML) provides:
      - Params/metrics visible in ZenML dashboard + Vertex AI console
      - TensorBoard integration via Vertex AI TensorBoard instance

    TensorBoardLogger provides:
      - Detailed per-step training curves, gradient histograms, embeddings
    """
    # 1. TensorBoard logger for detailed training curves → GCS
    tb_log_dir = config.training.tensorboard_log_dir
    tb_logger = TensorBoardLogger(
        save_dir=tb_log_dir,
        name=config.get("experiment_name", "headway_tft"),
        default_hp_metric=False,
        log_graph=True,
    )
    logger.info("TensorBoard logging to: %s", tb_log_dir)

    # 2. Log hyperparameters to Vertex AI Experiments (shows in ZenML + GCP)
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
    }
    aiplatform.log_params(hparams)
    tb_logger.log_hyperparams(hparams)

    # 3. Train
    result = train_tft(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=config,
        lightning_logger=tb_logger,
    )

    # 4. Log final metric to Vertex AI Experiments
    aiplatform.log_metrics({"best_val_loss": result.best_val_loss})

    return result.model
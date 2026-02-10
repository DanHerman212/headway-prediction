"""train_model.py — ZenML training step using TensorBoard for experiment tracking."""

import logging

from lightning.pytorch.loggers import TensorBoardLogger
from zenml import step
from omegaconf import DictConfig, OmegaConf
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from ..training_core import train_tft

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def train_model_step(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig,
) -> TemporalFusionTransformer:
    """
    Train the TFT model with TensorBoard logging.

    Logs training scalars, gradient histograms, and hyperparameters
    to the GCS-backed TensorBoard log directory defined in config.
    """
    # 1. Set up TensorBoard logger → writes to GCS
    tb_log_dir = config.training.tensorboard_log_dir
    tb_logger = TensorBoardLogger(
        save_dir=tb_log_dir,
        name=config.get("experiment_name", "headway_tft"),
        default_hp_metric=False,
    )
    logger.info("TensorBoard logging to: %s", tb_log_dir)

    # 2. Log hyperparameters to TensorBoard HParams tab
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
    tb_logger.log_hyperparams(hparams)

    # 3. Execute Training
    result = train_tft(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=config,
        lightning_logger=tb_logger,
    )

    # 4. Return the trained model
    return result.model
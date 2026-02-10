"""
training_core.py
----------------
Reusable training logic shared by both the production training pipeline
and the HPO trial entrypoint. This ensures that both the main pipeline
and the HPO usage run identical training procedures.

No ZenML or MLflow dependencies should reside here — this module is
pure PyTorch Lightning + PyTorch Forecasting.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from omegaconf import DictConfig, OmegaConf
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from .model_definitions import create_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Container for training outputs consumed by callers."""
    model: TemporalFusionTransformer
    best_val_loss: float


def train_tft(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig,
    lightning_logger: Optional[Logger] = None,
) -> TrainingResult:
    """
    Core training loop for the Temporal Fusion Transformer.

    Parameters
    ----------
    training_dataset : TimeSeriesDataSet
        Processed training split.
    validation_dataset : TimeSeriesDataSet
        Processed validation split.
    config : DictConfig
        Hydra config containing ``model.*`` and ``training.*`` keys.
    lightning_logger : Logger, optional
        A Lightning logger instance (MLFlowLogger, TensorBoardLogger, etc.).
        If None, falls back to a local CSVLogger.

    Returns
    -------
    TrainingResult
        The trained model and the best validation loss achieved.
    """
    # 1. Create DataLoaders
    #    Note: 'num_workers' and 'pin_memory' are critical for performance
    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        # Validation batches can be larger since no backprop is needed
        batch_size=config.training.batch_size * config.training.val_batch_size_multiplier,
        num_workers=config.training.num_workers,
    )

    # 2. Create Model
    tft = create_model(training_dataset, config)

    # 3. Define Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=config.training.early_stopping_min_delta,
        patience=config.training.early_stopping_patience,
        mode="min",
    )
    lr_monitor = LearningRateMonitor()

    # 4. Handle Logger
    if lightning_logger is None:
        # Fallback — TensorBoardLogger handles add_embedding/add_histogram
        # that pytorch-forecasting calls on logger.experiment.
        # CSVLogger does NOT support these and will crash.
        lightning_logger = TensorBoardLogger(save_dir=".", name="training_logs")

    # 5. Initialize Trainer
    #    Optional PyTorch profiler for GPU kernel traces in TensorBoard
    profiler = None
    profiler_name = OmegaConf.select(config, "training.profiler", default=None)
    if profiler_name == "pytorch":
        from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
        sched_cfg = OmegaConf.select(config, "training.profiler_schedule", default={})
        profiler = pl.profilers.PyTorchProfiler(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=sched_cfg.get("wait", 1),
                warmup=sched_cfg.get("warmup", 1),
                active=sched_cfg.get("active", 3),
                repeat=sched_cfg.get("repeat", 2),
            ),
            on_trace_ready=tensorboard_trace_handler(
                str(lightning_logger.log_dir) if hasattr(lightning_logger, "log_dir") else "./profiler_logs"
            ),
            record_shapes=True,
            with_stack=False,
        )
        logger.info("PyTorch profiler enabled — traces will appear in TensorBoard Profile tab")

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        enable_model_summary=config.training.enable_model_summary,
        gradient_clip_val=config.training.gradient_clip_val,
        limit_train_batches=config.training.limit_train_batches,
        precision=config.training.precision,
        callbacks=[lr_monitor, early_stop_callback],
        logger=lightning_logger,
        profiler=profiler,
    )

    # 6. Fit the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 7. Extract best metric
    #    The EarlyStopping callback tracks 'best_score' which corresponds to the monitored metric
    best_val_loss = float(early_stop_callback.best_score)
    logger.info(f"Training completed. Best val_loss: {best_val_loss:.6f}")

    return TrainingResult(model=tft, best_val_loss=best_val_loss)

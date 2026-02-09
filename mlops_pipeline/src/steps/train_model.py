import mlflow
import lightning.pytorch as pl
from zenml import step
from omegaconf import DictConfig
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from ..model_definitions import create_model

@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model_step(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig
) -> TemporalFusionTransformer:
    """
    Configures the Trainer and executes the training loop.
    Autologs metrics to MLflow.
    """
    # 1. Enable MLflow Autologging
    mlflow.pytorch.autolog()

    # 2. Create DataLoaders
    train_dataloader = training_dataset.to_dataloader(
        train=True, 
        batch_size=config.training.batch_size, 
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, 
        # Notebook used multiplier for val batch size
        batch_size=config.training.batch_size * config.training.val_batch_size_multiplier, 
        num_workers=config.training.num_workers
    )

    # 3. Create Model
    tft = create_model(training_dataset, config)

    # 4. Define Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=config.training.early_stopping_min_delta,
        patience=config.training.early_stopping_patience,
        mode="min"
    )
    lr_logger = LearningRateMonitor()

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        enable_model_summary=config.training.enable_model_summary,
        gradient_clip_val=config.training.gradient_clip_val,
        limit_train_batches=config.training.limit_train_batches,
        precision=config.training.precision,
        callbacks=[lr_logger, early_stop_callback],
    )

    # 6. Fit
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # 7. Return best model (reloaded from checkpoint automatically by BestModel logic if accessed, 
    #    but here we return the generic object. ZenML/MLflow handles artifact serialization).
    return tft
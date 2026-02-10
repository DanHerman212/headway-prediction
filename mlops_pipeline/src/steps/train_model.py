import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from zenml import step
from omegaconf import DictConfig
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from ..training_core import train_tft


class SafeMLFlowLogger(MLFlowLogger):
    """
    MLFlowLogger subclass that silently ignores TensorBoard-specific methods
    (add_embedding, add_histogram, add_figure, add_image, etc.) that
    pytorch-forecasting's BaseModel calls on logger.experiment.

    MLFlowLogger.experiment returns an MlflowClient, which lacks these methods.
    This wrapper intercepts the experiment property and returns a proxy that
    delegates known methods to MlflowClient and no-ops everything else.
    """

    class _ExperimentProxy:
        """Proxy that wraps MlflowClient and silently swallows unknown method calls."""

        def __init__(self, client):
            self._client = client

        def __getattr__(self, name):
            # If MlflowClient has the method, delegate to it
            if hasattr(self._client, name):
                return getattr(self._client, name)
            # Otherwise return a no-op callable (swallows any args/kwargs)
            return lambda *args, **kwargs: None

    @property
    def experiment(self):
        client = super().experiment
        return self._ExperimentProxy(client)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model_step(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: DictConfig,
) -> TemporalFusionTransformer:
    """
    Configures the Trainer and executes the training loop.
    Logs metrics to MLflow via Lightning's MLFlowLogger.

    Refactored to delegate core logic to src.training_core.train_tft
    """
    # 1. Get the active MLflow run (created by ZenML's experiment_tracker)
    active_run = mlflow.active_run()
    if active_run:
        mlflow_logger = SafeMLFlowLogger(
            experiment_name=active_run.info.experiment_id,
            run_id=active_run.info.run_id,
            tracking_uri=mlflow.get_tracking_uri(),
        )
    else:
        mlflow_logger = None

    # 2. Log training hyperparameters
    #    (This remains in the step because it's specific to MLflow experiment tracking)
    mlflow.log_params({
        "batch_size": config.training.batch_size,
        "max_epochs": config.training.max_epochs,
        "learning_rate": config.model.learning_rate,
        "gradient_clip_val": config.training.gradient_clip_val,
        "precision": config.training.precision,
        "accelerator": config.training.accelerator,
        "early_stopping_patience": config.training.early_stopping_patience,
        "hidden_size": config.model.hidden_size,
        "attention_head_size": config.model.attention_head_size,
        "dropout": config.model.dropout,
        "hidden_continuous_size": config.model.hidden_continuous_size,
    })

    # 3. Execute Training
    result = train_tft(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=config,
        lightning_logger=mlflow_logger,
    )

    # 4. Return the trained model
    return result.model
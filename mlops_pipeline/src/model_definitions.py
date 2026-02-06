import torch
from omegaconf import DictConfig
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import TimeSeriesDataSet

class TFTDisablePlotting(TemporalFusionTransformer):
    """
    Subclass of TFT that disables the log_prediction hook.
    This prevents Matplotlib crashes when using BFloat16 precision on A100s.
    """
    def log_prediction(self, x, out, batch_idx, **kwargs):
        # SKIP plotting during training to avoid BFloat16 Matplotlib crash or overhead
        pass

def create_model(training_dataset: TimeSeriesDataSet, config: DictConfig) -> TemporalFusionTransformer:
    """
    Initializes the TFT model using parameters from the Hydra configuration.
    """
    # Parse quantiles from config list
    quantiles = list(config.model.quantiles)
    
    # Initialize implementation class (with the plotting fix)
    tft = TFTDisablePlotting.from_dataset(
        training_dataset,
        learning_rate=config.model.learning_rate,
        hidden_size=config.model.hidden_size,
        lstm_layers=config.model.lstm_layers,
        attention_head_size=config.model.attention_head_size,
        dropout=config.model.dropout,
        hidden_continuous_size=config.model.hidden_continuous_size,
        output_size=len(quantiles),  # Number of quantiles
        loss=QuantileLoss(quantiles),
        log_interval=config.model.log_interval,
        reduce_on_plateau_patience=config.model.reduce_on_plateau_patience,
        optimizer=config.model.optimizer,
        reduce_on_plateau_min_lr=config.model.reduce_on_plateau_min_lr,
    )
    
    return tft
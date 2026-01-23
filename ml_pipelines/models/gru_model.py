"""
Stacked GRU Model for Multi-Output Headway Prediction

Architecture:
- Input: (batch, lookback_steps, n_features)
- Stacked GRU layers with configurable units
- Dual outputs:
  1. Headway prediction (regression)
  2. Route classification (multi-class)
"""

from typing import TYPE_CHECKING
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if TYPE_CHECKING:
    from config.model_config import ModelConfig

from .base_model import BaseModel


class StackedGRUModel(BaseModel):
    """
    Stacked GRU model for multi-output time series prediction.
    
    Features:
    - Multiple stacked GRU layers with configurable units
    - Dropout and recurrent dropout for regularization
    - Dual outputs for regression and classification
    
    Example:
        config = ModelConfig.from_env()
        model_builder = StackedGRUModel(config)
        model = model_builder.create()
    """
    
    def build(self) -> keras.Model:
        """
        Build the stacked GRU model architecture.
        
        Returns:
            Compiled Keras model with dual outputs
        """
        # Input layer: (batch, lookback_steps, n_features)
        # n_features = 8 (log_headway, day_of_week_sin/cos, hour_sin/cos, route_A/C/E)
        inputs = layers.Input(
            shape=(self.config.lookback_steps, 8),
            name='input_sequence'
        )
        
        # First GRU layer
        # Return sequences=True to pass to next GRU layer
        gru1 = layers.GRU(
            units=self.config.gru_units[0],
            return_sequences=True,
            name='gru_1'
        )(inputs)
        
        # Second GRU layer
        # Return sequences=False to output single vector for prediction
        gru2 = layers.GRU(
            units=self.config.gru_units[1],
            return_sequences=False,
            name='gru_2'
        )(gru1)
        
        # Output 1: Headway prediction (regression)
        # Single value: log_headway
        headway_output = layers.Dense(
            units=1,
            activation=None,  # Linear activation for regression
            name='headway'
        )(gru2)
        
        # Output 2: Route classification (multi-class)
        # n_routes classes (A, C, E)
        route_output = layers.Dense(
            units=self.config.n_routes,
            activation='softmax',
            name='route'
        )(gru2)
        
        # Create model with dual outputs
        model = keras.Model(
            inputs=inputs,
            outputs=[headway_output, route_output],
            name='stacked_gru_multi_output'
        )
        
        return model
    
    def get_architecture_summary(self) -> str:
        """
        Get a human-readable summary of the architecture.
        
        Returns:
            Architecture description string
        """
        summary = f"""
Stacked GRU Multi-Output Model
{'='*60}
Architecture:
  Input: ({self.config.lookback_steps} timesteps, 8 features)
  ↓
  GRU({self.config.gru_units[0]}) → return_sequences=True
  ↓
  GRU({self.config.gru_units[1]}) → return_sequences=False
  ↓
  ├─→ Dense(1) → Headway (regression)
  └─→ Dense({self.config.n_routes}, softmax) → Route (classification)

Outputs:
  1. Headway: Log-transformed headway prediction
  2. Route: Route classification (A/C/E)

Parameters:
  - Lookback steps: {self.config.lookback_steps}
  - GRU units: {self.config.gru_units}
  - Total routes: {self.config.n_routes}
  - No regularization (initial overfitting test)
"""
        return summary

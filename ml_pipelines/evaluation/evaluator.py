"""
Model Evaluator

Comprehensive model evaluation with visualizations and analysis.
"""

import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ml_pipelines.config import ModelConfig


class Evaluator:
    """
    Comprehensive model evaluation and visualization.
    
    Features:
    - Training curve visualization
    - Test set evaluation
    - Prediction analysis
    - Error distribution analysis
    - Model performance summaries
    
    Example:
        evaluator = Evaluator(
            model=trained_model,
            config=model_config,
            scaler=data_scaler
        )
        
        # Plot training curves
        evaluator.plot_training_curves(history, save_path="training_curves.png")
        
        # Evaluate on test set
        test_metrics = evaluator.evaluate(test_dataset)
        
        # Generate predictions
        predictions = evaluator.predict(test_dataset)
        
        # Plot predictions vs actual
        evaluator.plot_predictions(y_true, y_pred, save_path="predictions.png")
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        config: ModelConfig,
        scaler: Optional[Any] = None,
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            config: ModelConfig instance
            scaler: Optional fitted scaler for inverse transform
            output_dir: Directory for saving evaluation outputs
        """
        self.model = model
        self.config = config
        self.scaler = scaler
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine data range from scaler or use defaults
        if scaler is not None and hasattr(scaler, 'data_min_'):
            self.data_min = scaler.data_min_[0]
            self.data_max = scaler.data_max_[0]
        else:
            self.data_min = 0.0
            self.data_max = 30.0  # Default: 30 minutes max
        
        self.data_range = self.data_max - self.data_min
        
        print(f"✓ Evaluator initialized")
        print(f"  Output dir: {output_dir}")
        print(f"  Data range: [{self.data_min:.2f}, {self.data_max:.2f}]")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        verbose: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: tf.data.Dataset to evaluate
            verbose: Verbosity mode
            
        Returns:
            Dictionary of metric names to values
        """
        print(f"\nEvaluating model...")
        results = self.model.evaluate(dataset, verbose=verbose, return_dict=True)
        
        print(f"\nEvaluation Results:")
        for name, value in results.items():
            print(f"  {name}: {value:.6f}")
        
        return results
    
    def predict(
        self,
        dataset: tf.data.Dataset,
        verbose: int = 0
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            dataset: tf.data.Dataset for prediction
            verbose: Verbosity mode
            
        Returns:
            Predictions as numpy array
        """
        predictions = self.model.predict(dataset, verbose=verbose)
        return predictions
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot_training_curves(
        self,
        history: tf.keras.callbacks.History,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Plot training and validation curves.
        
        Args:
            history: Keras History object from training
            save_path: Path to save figure (None = display only)
            figsize: Figure size
        """
        history_dict = history.history
        epochs = range(1, len(history_dict['loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss curve
        ax = axes[0]
        ax.plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history_dict:
            ax.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Metric curve (if available)
        ax = axes[1]
        # Find first non-loss metric
        metric_key = None
        for key in history_dict.keys():
            if key not in ['loss', 'val_loss'] and not key.startswith('val_'):
                metric_key = key
                break
        
        if metric_key:
            val_metric_key = f'val_{metric_key}'
            ax.plot(epochs, history_dict[metric_key], 'b-', label=f'Training {metric_key}', linewidth=2)
            if val_metric_key in history_dict:
                ax.plot(epochs, history_dict[val_metric_key], 'r-', label=f'Validation {metric_key}', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric_key, fontsize=11)
            ax.set_title(f'{metric_key} Over Time', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = os.path.join(self.output_dir, save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Training curves saved to: {save_path}")
        
        plt.close()
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        num_samples: int = 100,
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save figure
            num_samples: Number of samples to plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Time series comparison
        ax = axes[0]
        indices = range(min(num_samples, len(y_true)))
        ax.plot(indices, y_true[:num_samples], 'b-', label='Actual', linewidth=2, alpha=0.7)
        ax.plot(indices, y_pred[:num_samples], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter plot
        ax = axes[1]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=11)
        ax.set_ylabel('Predicted Values', fontsize=11)
        ax.set_title('Prediction Scatter Plot', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = os.path.join(self.output_dir, save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Prediction plot saved to: {save_path}")
        
        plt.close()
    
    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Plot error distribution analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save figure
            figsize: Figure size
        """
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Error histogram
        ax = axes[0]
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Error vs actual
        ax = axes[1]
        ax.scatter(y_true, errors, alpha=0.5, s=20)
        ax.axhline(0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual Values', fontsize=11)
        ax.set_ylabel('Prediction Error', fontsize=11)
        ax.set_title('Error vs Actual', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Absolute error vs actual
        ax = axes[2]
        abs_errors = np.abs(errors)
        ax.scatter(y_true, abs_errors, alpha=0.5, s=20)
        ax.set_xlabel('Actual Values', fontsize=11)
        ax.set_ylabel('Absolute Error', fontsize=11)
        ax.set_title('Absolute Error vs Actual', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = os.path.join(self.output_dir, save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Error distribution saved to: {save_path}")
        
        plt.close()
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def to_real_units(self, normalized_value: np.ndarray) -> np.ndarray:
        """
        Convert normalized values to real units.
        
        Args:
            normalized_value: Normalized values
            
        Returns:
            Values in real units
        """
        return normalized_value * self.data_range + self.data_min
    
    def to_normalized(self, real_value: np.ndarray) -> np.ndarray:
        """
        Convert real values to normalized units.
        
        Args:
            real_value: Real values
            
        Returns:
            Normalized values
        """
        return (real_value - self.data_min) / self.data_range

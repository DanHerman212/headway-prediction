"""
evaluate_model.py
-----------------
Evaluation step for the Headway Prediction Pipeline.
Calculates global metrics (MAE, SMAPE) and generates 'Rush Hour' specific plots for A, C, E lines.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, Any

import mlflow
from zenml import step
from omegaconf import DictConfig
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RushHourVisualizer:
    """
    Helper class to generate specific Rush Hour performance plots for A, C, E lines.
    """
    def __init__(self, predictions: torch.Tensor, x: Dict[str, torch.Tensor], group_encoder: Any):
        self.predictions = predictions.cpu() # Shape: (Batch, PredictionLength, Quantiles)
        self.x = x # Dictionary containing decoder_target, groups, time_idx
        self.group_encoder = group_encoder
        
    def _reconstruct_dataframe(self) -> pd.DataFrame:
        """
        Reconstructs a flat DataFrame from the Tensor outputs for easier plotting.
        """
        # 1. Extract Targets and Predictions
        actuals = self.x["decoder_target"].cpu().view(-1).numpy()
        
        # We use the P50 (Median) for the main prediction line
        # prediction shape is (Batch, Time, Quantiles), P50 is usually index 1
        p50 = self.predictions[:, :, 1].view(-1).numpy()
        p10 = self.predictions[:, :, 0].view(-1).numpy()
        p90 = self.predictions[:, :, 2].view(-1).numpy()
        
        # 2. Extract Groups
        # Groups in x['groups'] are (Batch, 1). We need to repeat them for sequence length.
        batch_groups = self.x["groups"].cpu().view(-1)
        seq_len = self.predictions.shape[1]
        
        # Repeat group_ids for every timestep in the sequence
        group_ids = batch_groups.repeat_interleave(seq_len).numpy()
        
        # Decode Group IDs to Strings (e.g., "0" -> "A", "1" -> "C")
        try:
            # Check if group_encoder has inverse_transform (LabelEncoder-like)
            if hasattr(self.group_encoder, "inverse_transform"):
                decoded_groups = self.group_encoder.inverse_transform(group_ids)
            elif hasattr(self.group_encoder, "classes_"):
                 # Manual mapping for some scikit-learn encoders if inverse_transform acts up with tensors
                 unique_ids = np.unique(group_ids)
                 mapping = {uid: self.group_encoder.classes_[uid] for uid in unique_ids}
                 decoded_groups = np.array([mapping[g] for g in group_ids])
            else:
                 # Fallback
                 decoded_groups = group_ids.astype(str)
        except Exception as e:
            logger.warning(f"Could not decode groups: {e}")
            decoded_groups = group_ids.astype(str)

        # 3. Create DataFrame
        # Verify if decoder_time_idx exists, else construct it from scratch or decoder_time_idx provided by PF
        if "decoder_time_idx" in self.x:
            batch_time_idx = self.x["decoder_time_idx"].cpu().view(-1) 
            time_idx = batch_time_idx.numpy()
        else:
             # Fallback: Create dummy index
             time_idx = np.arange(len(actuals))

        df = pd.DataFrame({
            "group": decoded_groups,
            "time_idx": time_idx,
            "actual": actuals,
            "pred_p50": p50,
            "pred_p10": p10,
            "pred_p90": p90
        })
        
        return df

    def plot_rush_hour(self, start_idx_window: int = None, window_size: int = 180) -> plt.Figure:
        """
        Plots a 'Rush Hour' window.
        Args:
            window_size: 180 minutes (3 hours)
        """
        df = self._reconstruct_dataframe()
        
        # Filter for A, C, E specifically
        target_groups = ['A', 'C', 'E']
        available_groups = df['group'].unique()
        
        # Normalize to handle potential variations like "Line A" vs "A"
        plot_groups = [g for g in target_groups if g in available_groups]
        
        # If none found (e.g. maybe encoded as numbers), fallback to top 3 busiest
        if not plot_groups:
            logger.warning(f"Target groups {target_groups} not found in {available_groups}. Using Top 3.")
            plot_groups = list(df['group'].value_counts().index[:3])
            
        fig, axes = plt.subplots(len(plot_groups), 1, figsize=(15, 5 * len(plot_groups)), sharex=False)
        if len(plot_groups) == 1:
            axes = [axes]
            
        for i, group in enumerate(plot_groups):
            ax = axes[i]
            group_df = df[df['group'] == group].sort_values("time_idx")
            
            if group_df.empty:
                continue

            # Heuristic: Find a busy 3-hour window
            # If start_idx_window not provided, pick the middle of the dataset
            if start_idx_window is None:
                mid_point = group_df['time_idx'].median()
                start_time = mid_point - (window_size / 2)
            else:
                start_time = start_idx_window

            end_time = start_time + window_size
            
            # Slice
            mask = (group_df['time_idx'] >= start_time) & (group_df['time_idx'] <= end_time)
            plot_df = group_df[mask]
            
            if plot_df.empty:
                ax.text(0.5, 0.5, "No Data in Window", ha='center')
                continue

            # Plotting
            ax.plot(plot_df['time_idx'], plot_df['actual'], 'ko', label='Actual', markersize=4, alpha=0.7)
            # Use specific colors for lines (MTA Blue #0039A6)
            ax.plot(plot_df['time_idx'], plot_df['pred_p50'], linestyle='-', label='Predicted', linewidth=2, color='#0039A6') 
            ax.fill_between(plot_df['time_idx'], plot_df['pred_p10'], plot_df['pred_p90'], color='#0039A6', alpha=0.15, label='90% CI')
            
            ax.set_title(f"Subway Line {group}: Rush Hour Trace", fontsize=14, fontweight='bold')
            ax.set_ylabel("Headway (min)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        ax.set_xlabel("Time Index (Minutes)")
        plt.tight_layout()
        return fig

@step(enable_cache=False)
def evaluate_model(
    model: TemporalFusionTransformer,
    test_dataset: TimeSeriesDataSet,
    config: DictConfig
) -> Tuple[float, float]:
    """
    Evaluates the model on the test set.
    """
    logger.info("Starting Model Evaluation on Test Set...")

    # 0. Create DataLoader from Dataset
    # We use validation batch size multiplier (usually larger than train bc no backprop)
    batch_size = config.training.batch_size * config.training.val_batch_size_multiplier
    
    test_loader = test_dataset.to_dataloader(
        train=False, 
        batch_size=batch_size, 
        num_workers=config.training.num_workers
    )
    
    # 1. Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # 2. Generate Predictions
    raw_prediction = model.predict(test_loader, mode="raw", return_x=True)
    
    predictions = raw_prediction.output["prediction"]
    x = raw_prediction.x
    
    # 3. Calculate Global Metrics
    actuals = x["decoder_target"].cpu()
    predictions_cpu = predictions.cpu()
    
    # P50 (Median) for point metrics
    p50_forecast = predictions_cpu[:, :, 1]
    
    mae_metric = MAE()
    smape_metric = SMAPE()
    
    mae = mae_metric(p50_forecast, actuals).item()
    smape = smape_metric(p50_forecast, actuals).item()
    
    logger.info(f"Global Test Metrics: MAE={mae:.4f}, sMAPE={smape:.4f}")
    
    # 4. Generate Rush Hour Visualizations
    logger.info("Generating Rush Hour Visualization...")
    
    if hasattr(model.dataset_parameters, "categorical_encoders"):
         group_encoder = model.dataset_parameters["categorical_encoders"]["group_id"]
    else:
         # Best effort recovery
         group_encoder = None

    viz = RushHourVisualizer(predictions, x, group_encoder)
    
    # Generate the Rush Hour plot
    fig = viz.plot_rush_hour(window_size=180) # 3 hour window
    
    # 5. Log to MLflow
    # Check if run is active (it should be in ZenML pipeline if properly integrated, or manual)
    try:
        # Check if an active run exists; if not, ZenML's MLflow integration usually handles it,
        # but explicit logging is safer within the step context if integration is enabled.
        if mlflow.active_run():
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_smape", smape)
            mlflow.log_figure(fig, "rush_hour_performance.png")
            logger.info("Evaluation metrics and plots logged to MLflow.")
        else:
            logger.warning("No active MLflow run detected. Skipping log_figure.")
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")
        # Save locally for debugging
        fig.savefig("rush_hour_performance_debug.png")
        logger.info("Saved rush_hour_performance_debug.png locally")
        
    plt.close(fig)
    
    return mae, smape

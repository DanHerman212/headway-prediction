"""                                                                              
evaluate_model.py                                                                 
-----------------                                                                 
Evaluation step for the Headway Prediction Pipeline.                               
Calculates global metrics (MAE, SMAPE) and generates 'Rush Hour' specific plots.   
Logs results to Vertex AI Experiments + TensorBoard.                               
"""

import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
import tempfile
from typing import Tuple, Dict, Any, Optional

from google.cloud import aiplatform
from torch.utils.tensorboard import SummaryWriter
from zenml import step, get_step_context
from omegaconf import DictConfig
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RushHourVisualizer:
    """
    Helper class to generate specific Rush Hour performance plots for A, C, E lines.
    Now matches 'Next Steps' spec: Plotly interactive, train_id titles, improved colors.
    """
    def __init__(self, predictions: torch.Tensor, x: Dict[str, torch.Tensor], group_encoder: Any):
        self.predictions = predictions.cpu() # Shape: (Batch, PredictionLength, Quantiles)
        self.x = x # Dictionary containing decoder_target, groups, time_idx
        self.group_encoder = group_encoder
        
    def _reconstruct_dataframe(self) -> pd.DataFrame:
        """
        Reconstructs a flat DataFrame from the Tensor outputs.
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
        
        # Decode Group IDs to Strings (e.g., "A_South_Train123")
        try:
            if hasattr(self.group_encoder, "inverse_transform"):
                decoded_groups = self.group_encoder.inverse_transform(group_ids)
            elif hasattr(self.group_encoder, "classes_"):
                 classes = self.group_encoder.classes_
                 if isinstance(classes, dict):
                     inv_map = {v: k for k, v in classes.items()}
                 else:
                     inv_map = {i: c for i, c in enumerate(classes)}
                 decoded_groups = np.array([inv_map.get(g, str(g)) for g in group_ids])
            else:
                 decoded_groups = group_ids.astype(str)
        except Exception as e:
            logger.warning(f"Could not decode groups: {e}")
            decoded_groups = group_ids.astype(str)

        # 3. Create DataFrame
        if "decoder_time_idx" in self.x:
            batch_time_idx = self.x["decoder_time_idx"].cpu().view(-1) 
            time_idx = batch_time_idx.numpy()
        else:
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

    def plot_rush_hour(self, start_idx_window: Optional[int] = None, window_size: int = 180) -> go.Figure:
        """
        Generates an interactive Plotly chart with subplots for specific trains/lines.
        
        Specs:
        - Subplot titles include train_id/group
        - X-axis in generic "Minutes from Start" (until we pass real timestamps)
        - Color scheme: MTA Blue for predictions, accessible contrast
        """
        df = self._reconstruct_dataframe()
        
        # Filter for A, C, E lines or reasonable fallback
        target_lines = ['A', 'C', 'E']
        available_groups = df['group'].unique()
        
        plot_groups = []
        for g in available_groups:
            line_letter = str(g).split('_')[0]
            if line_letter in target_lines:
                plot_groups.append(g)
        
        # Limit to 3 distinct groups to keep plot readable
        if not plot_groups:
            plot_groups = list(available_groups)[:3]
        else:
            plot_groups = plot_groups[:3]
            
        # Create Subplots
        fig = make_subplots(
            rows=len(plot_groups), cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=[f"Train: {g}" for g in plot_groups]
        )

        for i, group in enumerate(plot_groups):
            row = i + 1
            group_df = df[df['group'] == group].sort_values("time_idx")
            
            if group_df.empty:
                continue

            # Heuristic: Find a busy window
            if start_idx_window is None:
                mid_point = group_df['time_idx'].median()
                start_time = mid_point - (window_size / 2)
            else:
                start_time = start_idx_window

            end_time = start_time + window_size
            
            # Slice the window
            mask = (group_df['time_idx'] >= start_time) & (group_df['time_idx'] <= end_time)
            plot_df = group_df[mask]
            
            if plot_df.empty:
                continue

            # 1. Prediction Interval (P10-P90) - Filled Area
            # Plotly trick: P10 (transparent line) + P90 (filled down to P10)
            fig.add_trace(go.Scatter(
                x=plot_df['time_idx'], y=plot_df['pred_p10'],
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_df['time_idx'], y=plot_df['pred_p90'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0, 57, 166, 0.2)', # MTA Blue transparent
                name='90% Confidence' if i == 0 else None,
                showlegend=(i == 0)
            ), row=row, col=1)

            # 2. Main Prediction (P50)
            fig.add_trace(go.Scatter(
                x=plot_df['time_idx'], y=plot_df['pred_p50'],
                mode='lines',
                line=dict(color='#0039A6', width=2), # MTA Blue
                name='Predicted (P50)' if i == 0 else None,
                showlegend=(i == 0)
            ), row=row, col=1)

            # 3. Actuals
            fig.add_trace(go.Scatter(
                x=plot_df['time_idx'], y=plot_df['actual'],
                mode='markers',
                marker=dict(color='black', size=5, symbol='circle'),
                name='Actual Headway' if i == 0 else None,
                showlegend=(i == 0)
            ), row=row, col=1)

            # Layout tweaks per subplot
            fig.update_xaxes(title_text="Time Index (Minutes)", row=row, col=1)
            fig.update_yaxes(title_text="Headway (min)", row=row, col=1)

        # Global Layout
        fig.update_layout(
            height=300 * len(plot_groups),
            title_text="Rush Hour Headway Predictions (Model Eval)",
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig

@step(enable_cache=False)
def evaluate_model(
    model: TemporalFusionTransformer,
    test_dataset: TimeSeriesDataSet,
    config: DictConfig
) -> Tuple[float, float]:
    """
    Evaluates the model on the test set.

    Logs metrics to the same Vertex AI Experiment run that the training step
    created, by resuming the run via aiplatform.start_run(resume=True).
    This avoids the TensorBoard run name collision that occurs when two steps
    both declare experiment_tracker="vertex_tracker".
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
    
    # 5. Log to Vertex AI Experiments by resuming the training step's run.
    #    The experiment name comes from config; the run name is the ZenML pipeline run ID
    #    (sanitized to match Vertex AI naming: lowercase, hyphens only).
    try:
        context = get_step_context()
        run_name = context.pipeline_run.name
    except Exception:
        run_name = None
        logger.warning("Could not retrieve ZenML step context for run name.")

    experiment_name = config.get("experiment_name", "headway_tft").lower().replace("_", "-")

    try:
        aiplatform.init(
            project=config.infra.project_id,
            location=config.infra.location,
            experiment=experiment_name,
        )
        if run_name:
            aiplatform.start_run(run_name, resume=True)
            aiplatform.log_metrics({"test_mae": mae, "test_smape": smape})
            aiplatform.end_run()
            logger.info("Logged eval metrics to Vertex AI Experiment run: %s", run_name)
        else:
            logger.warning("No run name available — skipping Vertex AI metric logging.")
    except Exception as e:
        logger.warning("Failed to log eval metrics to Vertex AI Experiments: %s", e)

    # 6. Log to TensorBoard (detailed plots + scalars) and save artifacts
    local_eval_dir = tempfile.mkdtemp(prefix="eval_tb_")
    
    try:
        # Save plots to temp dir (not container CWD which is lost on exit)
        html_path = os.path.join(local_eval_dir, "rush_hour_performance.html")
        png_path = os.path.join(local_eval_dir, "rush_hour_performance.png")
        
        # Save HTML (Interactive)
        fig.write_html(html_path)
        logger.info(f"Saved interactive plot to {html_path}")
        
        # Save PNG (Static) - requires kaleido
        has_png = False
        try:
             fig.write_image(png_path, scale=2)
             has_png = True
             logger.info(f"Saved static plot to {png_path}")
        except Exception as e:
             logger.warning(f"Could not save static PNG (requires kaleido): {e}")

        # Log scalars + image to TensorBoard
        writer = SummaryWriter(log_dir=local_eval_dir)
        writer.add_scalar("eval/test_mae", mae, global_step=0)
        writer.add_scalar("eval/test_smape", smape, global_step=0)
        
        # Log rush hour plot as TensorBoard image if PNG was generated
        if has_png:
            try:
                from PIL import Image
                img = Image.open(png_path)
                img_array = np.array(img)
                # TensorBoard expects (H, W, C) for add_image with dataformats
                writer.add_image("eval/rush_hour_performance", img_array, global_step=0, dataformats='HWC')
                logger.info("Logged rush hour plot to TensorBoard as image")
            except Exception as img_err:
                logger.warning(f"Could not log image to TensorBoard: {img_err}")
        
        writer.close()

        # Upload everything (TB events + HTML + PNG) to GCS
        gcs_tb_dir = config.training.tensorboard_log_dir
        from google.cloud import storage as gcs_storage
        try:
            bucket_name = gcs_tb_dir.replace("gs://", "").split("/")[0]
            prefix = "/".join(gcs_tb_dir.replace("gs://", "").split("/")[1:])
            client = gcs_storage.Client()
            bucket = client.bucket(bucket_name)
            for root, _, files in os.walk(local_eval_dir):
                for fname in files:
                    local_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(local_path, local_eval_dir)
                    blob_path = f"{prefix}/evaluation/{rel_path}"
                    bucket.blob(blob_path).upload_from_filename(local_path)
            logger.info("Uploaded evaluation artifacts to %s/evaluation", gcs_tb_dir)
        except Exception as upload_err:
            logger.warning(f"Failed to upload evaluation artifacts: {upload_err}")
        
    except Exception as e:
        logger.warning(f"Error during artifact saving: {e}")
    finally:
        import shutil
        shutil.rmtree(local_eval_dir, ignore_errors=True)
    
    return mae, smape

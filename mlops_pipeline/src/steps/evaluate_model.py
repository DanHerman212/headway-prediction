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
    Generates Rush Hour performance plots for A, C, E subway lines.

    Requirements:
      - Subplot titles show the decoded group_id (e.g. "A_northbound_14st")
      - Every subplot has full hover labels (90% Confidence, Predicted P50, Actual)
      - X-axis shows real timestamps derived from time_idx + dataset time anchor
      - MTA Blue (#0039A6) colour scheme with accessible contrast
    """

    def __init__(
        self,
        predictions: torch.Tensor,
        x: Dict[str, torch.Tensor],
        test_dataset: TimeSeriesDataSet,
        time_anchor_iso: str = "",
    ):
        self.predictions = predictions.cpu()  # (Batch, PredLen, Quantiles)
        self.x = x
        self.test_dataset = test_dataset
        self.time_anchor = (
            pd.Timestamp(time_anchor_iso) if time_anchor_iso else None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decode_groups(self, encoded_ids: np.ndarray) -> np.ndarray:
        """Decode integer-encoded group IDs back to human-readable strings."""
        try:
            params = self.test_dataset.get_parameters()
            encoder = params.get("categorical_encoders", {}).get("group_id")
            if encoder is not None and hasattr(encoder, "inverse_transform"):
                return encoder.inverse_transform(encoded_ids)
        except Exception as e:
            logger.warning("Could not decode groups via get_parameters(): %s", e)

        # Fallback: use decoded_index which stores string group_ids per sample
        try:
            di = self.test_dataset.decoded_index
            if "group_id" in di.columns:
                unique_groups = di["group_id"].unique()
                return np.array([
                    unique_groups[int(g)] if int(g) < len(unique_groups) else str(g)
                    for g in encoded_ids
                ])
        except Exception as e:
            logger.warning("Fallback group decoding also failed: %s", e)

        return encoded_ids.astype(str)

    def _time_idx_to_timestamps(self, time_idx: np.ndarray) -> Optional[np.ndarray]:
        """Convert integer time_idx values to real datetime timestamps.

        time_idx is minutes elapsed from the global min arrival_time_dt
        (computed in data_processing.clean_dataset).  The anchor is passed
        in as ``time_anchor_iso``.
        """
        if self.time_anchor is None:
            return None
        try:
            timestamps = self.time_anchor + pd.to_timedelta(time_idx, unit="min")
            return timestamps.values
        except Exception as e:
            logger.warning("Could not convert time_idx to timestamps: %s", e)
            return None

    def _reconstruct_dataframe(self) -> pd.DataFrame:
        """Build a flat DataFrame from tensor outputs with decoded groups and timestamps."""
        actuals = self.x["decoder_target"].cpu().view(-1).numpy()
        p50 = self.predictions[:, :, 1].view(-1).numpy()
        p10 = self.predictions[:, :, 0].view(-1).numpy()
        p90 = self.predictions[:, :, 2].view(-1).numpy()

        batch_groups = self.x["groups"].cpu().view(-1)
        seq_len = self.predictions.shape[1]
        group_ids_encoded = batch_groups.repeat_interleave(seq_len).numpy()
        decoded_groups = self._decode_groups(group_ids_encoded)

        if "decoder_time_idx" in self.x:
            time_idx = self.x["decoder_time_idx"].cpu().view(-1).numpy()
        else:
            time_idx = np.arange(len(actuals))

        timestamps = self._time_idx_to_timestamps(time_idx)

        df = pd.DataFrame({
            "group": decoded_groups,
            "time_idx": time_idx,
            "actual": actuals,
            "pred_p50": p50,
            "pred_p10": p10,
            "pred_p90": p90,
        })
        if timestamps is not None:
            df["timestamp"] = timestamps

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_rush_hour(
        self,
        start_idx_window: Optional[int] = None,
        window_size: int = 180,
    ) -> go.Figure:
        """Generate an interactive Plotly figure with per-group subplots.

        Parameters
        ----------
        start_idx_window : int, optional
            Absolute time_idx to start the window.  If *None* the visualizer
            picks a window centred on the median time_idx for each group.
        window_size : int
            Width of the display window in minutes (default 180 = 3 hours).
        """
        df = self._reconstruct_dataframe()
        has_timestamps = "timestamp" in df.columns

        # ---- Select groups for A / C / E lines ----
        target_lines = ["A", "C", "E"]
        available_groups = df["group"].unique()

        plot_groups = []
        for g in available_groups:
            line_letter = str(g).split("_")[0]
            if line_letter in target_lines:
                plot_groups.append(g)

        if not plot_groups:
            plot_groups = list(available_groups)[:3]
        else:
            # One per target line, up to 3
            seen_lines: Dict[str, str] = {}
            for g in plot_groups:
                letter = str(g).split("_")[0]
                if letter not in seen_lines:
                    seen_lines[letter] = g
            plot_groups = list(seen_lines.values())[:3]

        # ---- Build subplots ----
        fig = make_subplots(
            rows=len(plot_groups),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.12,
            subplot_titles=[str(g) for g in plot_groups],
        )

        for i, group in enumerate(plot_groups):
            row = i + 1
            group_df = df[df["group"] == group].sort_values("time_idx")
            if group_df.empty:
                continue

            # --- Time window ---
            if start_idx_window is None:
                mid = group_df["time_idx"].median()
                t_start = mid - (window_size / 2)
            else:
                t_start = start_idx_window
            t_end = t_start + window_size

            plot_df = group_df[
                (group_df["time_idx"] >= t_start) & (group_df["time_idx"] <= t_end)
            ]
            if plot_df.empty:
                continue

            # Use timestamps for x-axis when available, raw time_idx otherwise
            x_vals = plot_df["timestamp"] if has_timestamps else plot_df["time_idx"]

            # --- Confidence band (P10 → P90) ---
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_df["pred_p10"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_df["pred_p90"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0, 57, 166, 0.2)",
                    name="90% Confidence",
                    legendgroup="confidence",
                    showlegend=(i == 0),
                    hovertemplate="P90: %{y:.2f}<extra>90% Confidence</extra>",
                ),
                row=row, col=1,
            )

            # --- P50 prediction ---
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_df["pred_p50"],
                    mode="lines",
                    line=dict(color="#0039A6", width=2),
                    name="Predicted (P50)",
                    legendgroup="p50",
                    showlegend=(i == 0),
                    hovertemplate="P50: %{y:.2f} min<extra>Predicted (P50)</extra>",
                ),
                row=row, col=1,
            )

            # --- Actuals ---
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_df["actual"],
                    mode="markers",
                    marker=dict(color="black", size=5, symbol="circle"),
                    name="Actual Headway",
                    legendgroup="actual",
                    showlegend=(i == 0),
                    hovertemplate="Actual: %{y:.2f} min<extra>Actual Headway</extra>",
                ),
                row=row, col=1,
            )

            # --- Axis labels ---
            x_title = "Time" if has_timestamps else "Time Index (minutes from dataset start)"
            fig.update_xaxes(title_text=x_title, row=row, col=1)
            fig.update_yaxes(title_text="Headway (min)", row=row, col=1)

            if has_timestamps:
                fig.update_xaxes(
                    tickformat="%b %d %H:%M",
                    dtick=30 * 60 * 1000,  # tick every 30 min (ms for datetime axes)
                    row=row, col=1,
                )

        fig.update_layout(
            height=350 * len(plot_groups),
            title_text="Rush Hour Headway Predictions — Model Evaluation (Test Set)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

@step(enable_cache=False)
def evaluate_model(
    model: TemporalFusionTransformer,
    test_dataset: TimeSeriesDataSet,
    config: DictConfig,
    time_anchor_iso: str = "",
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

    viz = RushHourVisualizer(predictions, x, test_dataset, time_anchor_iso)
    
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

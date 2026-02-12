"""
evaluate_model.py
-----------------
Evaluation step for the Headway Prediction Pipeline.

Calculates global metrics (MAE, sMAPE) and generates rush-hour performance
plots with strict data-quality constraints:

  * Only *weekday* data is used (Mon-Fri).
  * A single calendar date is chosen that has the best AM-rush (07:00-10:00)
    coverage across **all** groups — so every subplot shows the same day.
  * Every prediction in the rush-hour window is plotted (no arbitrary
    sub-sampling or window trimming).
  * Subplot titles include the decoded group_id, date, and period label.

All artefacts are consolidated into the Vertex AI Experiments UI:
  - Scalar metrics  → aiplatform.log_metrics()
  - TB events       → aiplatform.upload_tb_log() (rush-hour PNG + TFT
                       interpretation figures attached to the managed
                       TensorBoard instance)
  - Interactive HTML → returned as a ZenML named artifact
"""

import logging
import os
import re
import tempfile
from typing import Annotated, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from google.cloud import aiplatform
from omegaconf import DictConfig
from plotly.subplots import make_subplots
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE
from torch.utils.tensorboard import SummaryWriter
from zenml import step, get_step_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rush-hour plot helpers
# ---------------------------------------------------------------------------

# Target lines we want to display (A / C / E at 14 St)
_TARGET_LINES = ("A", "C", "E")

# AM Rush definition
_AM_RUSH_START = 7   # inclusive
_AM_RUSH_END = 10    # exclusive

# Minimum number of observations per group in the rush window for a date
# to be considered valid.  At ≥6-minute peak headways for trunk lines,
# 3 hours should yield ~30 trains.  Require at least 10 to avoid sparse
# dates that would misrepresent model performance.
_MIN_OBS_PER_GROUP = 10


def _decode_group_ids(
    encoded_ids: np.ndarray,
    test_dataset: TimeSeriesDataSet,
) -> np.ndarray:
    """Map integer-encoded group IDs back to human-readable strings."""
    try:
        params = test_dataset.get_parameters()
        encoder = params.get("categorical_encoders", {}).get("group_id")
        if encoder is not None and hasattr(encoder, "inverse_transform"):
            return encoder.inverse_transform(encoded_ids)
    except Exception as exc:
        logger.warning("get_parameters() group decode failed: %s", exc)

    # Fallback via decoded_index
    try:
        di = test_dataset.decoded_index
        if "group_id" in di.columns:
            unique = di["group_id"].unique()
            return np.array([
                unique[int(g)] if int(g) < len(unique) else str(g)
                for g in encoded_ids
            ])
    except Exception as exc:
        logger.warning("decoded_index fallback also failed: %s", exc)

    return encoded_ids.astype(str)


def _build_eval_dataframe(
    predictions: torch.Tensor,
    x: Dict[str, torch.Tensor],
    test_dataset: TimeSeriesDataSet,
    time_anchor: pd.Timestamp,
) -> pd.DataFrame:
    """Flatten model outputs into a tidy DataFrame with real timestamps.

    Columns: group, timestamp, date, hour, weekday, time_idx,
             actual, pred_p10, pred_p50, pred_p90
    """
    actuals = x["decoder_target"].cpu().view(-1).numpy()
    p10 = predictions[:, :, 0].cpu().view(-1).numpy()
    p50 = predictions[:, :, 1].cpu().view(-1).numpy()
    p90 = predictions[:, :, 2].cpu().view(-1).numpy()

    # Groups (repeat for each prediction step)
    batch_groups = x["groups"].cpu().view(-1)
    seq_len = predictions.shape[1]
    group_enc = batch_groups.repeat_interleave(seq_len).numpy()
    groups = _decode_group_ids(group_enc, test_dataset)

    # Time indices → timestamps
    if "decoder_time_idx" in x:
        time_idx = x["decoder_time_idx"].cpu().view(-1).numpy()
    else:
        time_idx = np.arange(len(actuals))

    timestamps = time_anchor + pd.to_timedelta(time_idx, unit="min")

    df = pd.DataFrame({
        "group": groups,
        "time_idx": time_idx,
        "timestamp": timestamps,
        "date": timestamps.date,
        "hour": timestamps.hour,
        "weekday": timestamps.dayofweek,     # 0=Mon … 4=Fri
        "actual": actuals,
        "pred_p10": p10,
        "pred_p50": p50,
        "pred_p90": p90,
    })
    return df


def _select_rush_date(
    df: pd.DataFrame,
    target_groups: List[str],
) -> Optional[pd.Timestamp]:
    """Choose the single best weekday date for AM rush-hour plotting.

    Selection criteria (in order):
      1. Weekday only (Mon-Fri).
      2. AM rush hours (07:00-10:00).
      3. Every *target_group* must have ≥ _MIN_OBS_PER_GROUP observations.
      4. Among qualifying dates, pick the one with the highest *minimum*
         observation count across groups (i.e. the date where coverage is
         most balanced).
    """
    rush = df[
        (df["weekday"] < 5) &
        (df["hour"] >= _AM_RUSH_START) &
        (df["hour"] < _AM_RUSH_END) &
        (df["group"].isin(target_groups))
    ]

    if rush.empty:
        logger.warning("No weekday AM rush data found for groups %s", target_groups)
        return None

    # Count observations per (date, group)
    counts = rush.groupby(["date", "group"]).size().unstack(fill_value=0)

    # Keep only dates where ALL target groups meet minimum threshold
    qualifying = counts[
        (counts[target_groups] >= _MIN_OBS_PER_GROUP).all(axis=1)
    ]

    if qualifying.empty:
        logger.warning(
            "No weekday AM rush date has ≥%d obs for every group.  "
            "Counts by date:\n%s",
            _MIN_OBS_PER_GROUP,
            counts.to_string(),
        )
        # Relax: pick the date with the highest total even if below threshold
        qualifying = counts

    # Best = highest minimum across groups (most balanced coverage)
    best_date = qualifying[target_groups].min(axis=1).idxmax()
    logger.info(
        "Selected rush-hour date: %s  (obs per group: %s)",
        best_date,
        counts.loc[best_date].to_dict() if best_date in counts.index else "N/A",
    )
    return best_date


def _select_target_groups(df: pd.DataFrame) -> List[str]:
    """Pick one group per target line (A, C, E) from available data."""
    available = df["group"].unique()
    chosen: Dict[str, str] = {}
    for g in sorted(available):
        letter = str(g).split("_")[0]
        if letter in _TARGET_LINES and letter not in chosen:
            chosen[letter] = g
    groups = [chosen[k] for k in sorted(chosen)]
    if not groups:
        # Fallback: take up to 3 groups
        groups = sorted(available)[:3]
    return groups


def build_rush_hour_figure(
    predictions: torch.Tensor,
    x: Dict[str, torch.Tensor],
    test_dataset: TimeSeriesDataSet,
    time_anchor_iso: str,
) -> go.Figure:
    """Build an interactive Plotly figure of AM rush-hour predictions.

    All groups are aligned to the same weekday date.  Every prediction
    in the 07:00-10:00 window on that date is included.
    """
    time_anchor = pd.Timestamp(time_anchor_iso)
    df = _build_eval_dataframe(predictions, x, test_dataset, time_anchor)

    target_groups = _select_target_groups(df)
    rush_date = _select_rush_date(df, target_groups)

    if rush_date is None:
        # Degenerate case: return an empty figure with an explanatory title
        fig = go.Figure()
        fig.update_layout(
            title_text="No qualifying weekday AM rush data found in the test set",
        )
        return fig

    date_label = pd.Timestamp(rush_date).strftime("%A %b %d, %Y")

    # Slice to rush window for selected date
    rush_df = df[
        (df["date"] == rush_date) &
        (df["hour"] >= _AM_RUSH_START) &
        (df["hour"] < _AM_RUSH_END) &
        (df["group"].isin(target_groups))
    ].sort_values(["group", "timestamp"])

    # Build subplots — one row per group
    n_groups = len(target_groups)
    fig = make_subplots(
        rows=n_groups,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"{g} — AM Rush ({_AM_RUSH_START:02d}:00-{_AM_RUSH_END:02d}:00) — {date_label}"
            for g in target_groups
        ],
    )

    for i, group in enumerate(target_groups):
        row = i + 1
        gdf = rush_df[rush_df["group"] == group]
        n_obs = len(gdf)

        if gdf.empty:
            continue

        x_vals = gdf["timestamp"]

        # Confidence band (P10 → P90)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=gdf["pred_p10"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=gdf["pred_p90"],
                mode="lines", line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 57, 166, 0.2)",
                name="90% Confidence",
                legendgroup="confidence",
                showlegend=(i == 0),
                hovertemplate="P90: %{y:.2f} min<extra>90% Confidence</extra>",
            ),
            row=row, col=1,
        )

        # P50 prediction line
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=gdf["pred_p50"],
                mode="lines",
                line=dict(color="#0039A6", width=2),
                name="Predicted (P50)",
                legendgroup="p50",
                showlegend=(i == 0),
                hovertemplate="P50: %{y:.2f} min<extra>Predicted (P50)</extra>",
            ),
            row=row, col=1,
        )

        # Actuals
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=gdf["actual"],
                mode="markers",
                marker=dict(color="black", size=5, symbol="circle"),
                name="Actual Headway",
                legendgroup="actual",
                showlegend=(i == 0),
                hovertemplate="Actual: %{y:.2f} min<extra>Actual Headway</extra>",
            ),
            row=row, col=1,
        )

        # Axis formatting
        fig.update_xaxes(
            title_text="Time" if row == n_groups else "",
            tickformat="%H:%M",
            dtick=15 * 60 * 1000,  # tick every 15 min
            row=row, col=1,
        )
        fig.update_yaxes(title_text="Headway (min)", row=row, col=1)

        # Annotate observation count in subplot title
        if i < len(fig.layout.annotations):
            existing = fig.layout.annotations[i].text
            fig.layout.annotations[i].update(
                text=f"{existing}  (n={n_obs})"
            )

    fig.update_layout(
        height=350 * n_groups,
        title_text=f"Rush Hour Headway Predictions — Test Set — {date_label}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )

    return fig

@step(enable_cache=False)
def evaluate_model(
    model: TemporalFusionTransformer,
    test_dataset: TimeSeriesDataSet,
    config: DictConfig,
    time_anchor_iso: str = "",
) -> Tuple[
    Annotated[float, "test_mae"],
    Annotated[float, "test_smape"],
    Annotated[str, "rush_hour_plot_html"],
]:
    """Evaluate the TFT model on the test set.

    Returns scalar metrics and an interactive rush-hour HTML plot.
    Logs everything to Vertex AI Experiments + managed TensorBoard.
    """
    logger.info("Starting model evaluation on test set …")

    # ── 1. Build dataloader and generate predictions ──────────────────
    batch_size = config.training.batch_size * config.training.val_batch_size_multiplier
    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    raw_prediction = model.predict(test_loader, mode="raw", return_x=True)
    raw_output = raw_prediction.output
    predictions = raw_output["prediction"]
    x = raw_prediction.x

    # ── 2. Global metrics (P50) ───────────────────────────────────────
    actuals = x["decoder_target"].cpu()
    predictions_cpu = predictions.cpu()
    p50_forecast = predictions_cpu[:, :, 1]

    mae = MAE()(p50_forecast, actuals).item()
    smape = SMAPE()(p50_forecast, actuals).item()
    logger.info("Global test metrics: MAE=%.4f  sMAPE=%.4f", mae, smape)

    # ── 3. Rush-hour figure ───────────────────────────────────────────
    logger.info("Building rush-hour visualisation …")
    fig = build_rush_hour_figure(predictions, x, test_dataset, time_anchor_iso)
    html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")

    # ── 4. Resolve run / experiment names ─────────────────────────────
    try:
        ctx = get_step_context()
        raw_name = ctx.pipeline_run.name
        run_name = re.sub(r"[^a-z0-9-]", "-", raw_name.strip().lower())[:128].rstrip("-")
    except Exception:
        run_name = None
        logger.warning("Could not resolve ZenML step context for run name.")

    experiment_name = (
        config.get("experiment_name", "headway_tft").lower().replace("_", "-")
    )
    tb_resource = (
        f"projects/{config.infra.project_id}/locations/{config.infra.location}"
        f"/tensorboards/{config.infra.tensorboard_instance_id}"
    )

    # ── 5a. Log scalar metrics to Vertex AI Experiments ───────────────
    try:
        aiplatform.init(
            project=config.infra.project_id,
            location=config.infra.location,
            experiment=experiment_name,
            experiment_tensorboard=tb_resource,
        )
        if run_name:
            aiplatform.start_run(run_name, resume=True)
            aiplatform.log_metrics({"test_mae": mae, "test_smape": smape})
            aiplatform.end_run()
            logger.info(
                "Logged eval metrics to Vertex AI Experiment run: %s", run_name,
            )
    except Exception as exc:
        logger.warning("Failed to log metrics to Vertex AI Experiments: %s", exc)

    # ── 5b. TensorBoard events → managed TensorBoard ─────────────────
    local_tb_dir = tempfile.mkdtemp(prefix="eval_tb_")
    try:
        writer = SummaryWriter(log_dir=local_tb_dir)
        writer.add_scalar("eval/test_mae", mae, global_step=0)
        writer.add_scalar("eval/test_smape", smape, global_step=0)

        # Rush-hour plot as PNG image
        try:
            png_path = os.path.join(local_tb_dir, "rush_hour.png")
            fig.write_image(png_path, scale=2)
            from PIL import Image

            img_arr = np.array(Image.open(png_path))
            writer.add_image(
                "eval/rush_hour_performance", img_arr,
                global_step=0, dataformats="HWC",
            )
        except Exception as img_exc:
            logger.warning("Could not add rush-hour PNG to TB: %s", img_exc)

        # TFT interpretation (feature importance + attention)
        try:
            logger.info("Generating TFT interpretation plots …")
            interpretation = model.interpret_output(raw_output, reduction="sum")
            interp_figs = model.plot_interpretation(interpretation)
            for key, interp_fig in interp_figs.items():
                writer.add_figure(
                    f"eval/interpretation_{key}", interp_fig, global_step=0,
                )
                plt.close(interp_fig)
            logger.info("Wrote %d interpretation figures to TB", len(interp_figs))
        except Exception as interp_exc:
            logger.warning("Interpretation plots failed: %s", interp_exc)

        writer.close()

        # Upload local TB events to managed TensorBoard instance
        tb_experiment_name = run_name or experiment_name
        aiplatform.upload_tb_log(
            tensorboard_id=config.infra.tensorboard_instance_id,
            tensorboard_experiment_name=tb_experiment_name,
            logdir=local_tb_dir,
            project=config.infra.project_id,
            location=config.infra.location,
            run_name_prefix="eval",
            description="Evaluation: rush-hour plot, TFT interpretation",
        )
        logger.info(
            "Uploaded eval TB events to managed TensorBoard (experiment=%s)",
            tb_experiment_name,
        )
    except Exception as exc:
        logger.warning("TensorBoard artifact upload failed: %s", exc)
    finally:
        import shutil
        shutil.rmtree(local_tb_dir, ignore_errors=True)

    return mae, smape, html_content

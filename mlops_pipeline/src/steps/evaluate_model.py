"""
evaluate_model.py
-----------------
Evaluation step for the Headway Prediction Pipeline.

Three responsibilities:
  1. Compute test MAE / sMAPE -> log to Vertex AI Experiments.
  2. Build a predictions-vs-actuals plot for a fixed time window
     (Jan 14 2026 15:00-21:00 ET, one subplot per route) -> return
     as an HTML ZenML artifact.
  3. Generate TFT interpretation plots (feature importance + attention)
     -> return as an HTML ZenML artifact.
"""

import io
import base64
import logging
import re
from typing import Annotated, Dict, Tuple

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
from zenml import step, get_step_context
from zenml.types import HTMLString

logger = logging.getLogger(__name__)

# -- Plot window (fixed) ---------------------------------------------------
PLOT_DATE = "2026-02-11"
PLOT_HOUR_START = 15  # inclusive
PLOT_HOUR_END = 20    # exclusive
PLOT_TZ = "America/New_York"


def _build_eval_dataframe(
    predictions: torch.Tensor,
    x: Dict[str, torch.Tensor],
    test_dataset: TimeSeriesDataSet,
    time_anchor: pd.Timestamp,
) -> pd.DataFrame:
    """Flatten model outputs + actuals into a DataFrame with real timestamps."""
    actuals = x["decoder_target"].cpu().view(-1).numpy()
    p10 = predictions[:, :, 0].cpu().view(-1).numpy()
    p50 = predictions[:, :, 1].cpu().view(-1).numpy()
    p90 = predictions[:, :, 2].cpu().view(-1).numpy()

    # Decode group_id ints -> strings
    enc = test_dataset.get_parameters()["categorical_encoders"]["group_id"]
    batch_groups = x["groups"].cpu().view(-1)
    seq_len = predictions.shape[1]
    group_enc = batch_groups.repeat_interleave(seq_len).numpy()
    groups = enc.inverse_transform(group_enc)

    # Derive route from group_id (e.g. "A_South" -> "A")
    routes = np.array([str(g).split("_")[0] for g in groups])

    # Reconstruct timestamps from time_idx
    time_idx = x["decoder_time_idx"].cpu().view(-1).numpy()
    timestamps = time_anchor + pd.to_timedelta(time_idx, unit="min")

    return pd.DataFrame({
        "group": groups,
        "route": routes,
        "timestamp": timestamps,
        "actual": actuals,
        "pred_p10": p10,
        "pred_p50": p50,
        "pred_p90": p90,
    })


def build_prediction_figure(df: pd.DataFrame) -> go.Figure:
    """Build a 3-subplot Plotly figure (one per route) for the fixed window."""
    df = df.copy()
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df["ts_et"] = ts.dt.tz_convert(PLOT_TZ)
    start = pd.Timestamp(f"{PLOT_DATE} {PLOT_HOUR_START:02d}:00", tz=PLOT_TZ)
    end = pd.Timestamp(f"{PLOT_DATE} {PLOT_HOUR_END:02d}:00", tz=PLOT_TZ)
    window = df[(df["ts_et"] >= start) & (df["ts_et"] < end)].sort_values("ts_et")

    routes = sorted(window["route"].unique())
    n = len(routes) or 1

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=False, vertical_spacing=0.10,
        subplot_titles=[
            f"{r} -- {PLOT_DATE} {PLOT_HOUR_START}:00-{PLOT_HOUR_END}:00 ET  "
            f"(n={len(window[window['route']==r])})"
            for r in routes
        ],
    )

    for i, route in enumerate(routes):
        row = i + 1
        rdf = window[window["route"] == route]
        x_vals = rdf["ts_et"]

        # Confidence band
        fig.add_trace(go.Scatter(
            x=x_vals, y=rdf["pred_p10"], mode="lines", line=dict(width=0),
            showlegend=False,
            name="P10 (lower)",
            hovertemplate="P10: %{y:.1f} min<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x_vals, y=rdf["pred_p90"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0,57,166,0.2)",
            name="P90 (upper)", legendgroup="ci", showlegend=(i == 0),
            hovertemplate="P90: %{y:.1f} min<extra></extra>",
        ), row=row, col=1)

        # P50
        fig.add_trace(go.Scatter(
            x=x_vals, y=rdf["pred_p50"], mode="lines",
            line=dict(color="#0039A6", width=2),
            name="Predicted (P50)", legendgroup="p50", showlegend=(i == 0),
            hovertemplate="P50: %{y:.1f} min<extra></extra>",
        ), row=row, col=1)

        # Actuals
        fig.add_trace(go.Scatter(
            x=x_vals, y=rdf["actual"], mode="markers",
            marker=dict(color="black", size=5),
            name="Actual", legendgroup="actual", showlegend=(i == 0),
            hovertemplate="Actual: %{y:.1f} min<extra></extra>",
        ), row=row, col=1)

        fig.update_xaxes(tickformat="%H:%M", dtick=30 * 60 * 1000, row=row, col=1)
        fig.update_yaxes(title_text="Headway (min)", row=row, col=1)

    fig.update_layout(
        height=300 * n,
        title_text=(
            f"Headway Predictions -- {PLOT_DATE} "
            f"{PLOT_HOUR_START}:00-{PLOT_HOUR_END}:00 ET"
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_interpretation_html(
    model: TemporalFusionTransformer,
    raw_output: Dict,
) -> str:
    """Render TFT interpretation plots as a self-contained HTML string."""
    interpretation = model.interpret_output(raw_output, reduction="sum")
    figs = model.plot_interpretation(interpretation)

    html_parts = ["<html><body><h1>TFT Interpretation</h1>"]
    for key, mpl_fig in figs.items():
        buf = io.BytesIO()
        mpl_fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(mpl_fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        html_parts.append(f"<h2>{key}</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{b64}"/>')
    html_parts.append("</body></html>")
    return "\n".join(html_parts)


@step(enable_cache=False)
def evaluate_model(
    model: TemporalFusionTransformer,
    test_dataset: TimeSeriesDataSet,
    config: DictConfig,
    time_anchor_iso: str = "",
) -> Tuple[
    Annotated[float, "test_mae"],
    Annotated[float, "test_smape"],
    Annotated[HTMLString, "rush_hour_plot_html"],
    Annotated[HTMLString, "interpretation_html"],
]:
    """Evaluate TFT on the test set.

    Returns
    -------
    test_mae, test_smape : float
        Global metrics on the full test set.
    rush_hour_plot_html : str
        Interactive Plotly HTML for the fixed prediction window.
    interpretation_html : str
        TFT feature-importance / attention plots as self-contained HTML.
    """
    logger.info("Starting evaluation ...")

    # -- 1. Predictions -----------------------------------------------------
    batch_size = config.training.batch_size * config.training.val_batch_size_multiplier
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=config.training.num_workers,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    raw_prediction = model.predict(test_loader, mode="raw", return_x=True)
    predictions = raw_prediction.output["prediction"]
    x = raw_prediction.x

    # -- 2. Metrics ---------------------------------------------------------
    actuals = x["decoder_target"].cpu()
    p50 = predictions.cpu()[:, :, 1]
    mae = MAE()(p50, actuals).item()
    smape = SMAPE()(p50, actuals).item()
    logger.info("Test MAE=%.4f  sMAPE=%.4f", mae, smape)

    # -- 3. Log metrics to Vertex AI Experiments ----------------------------
    try:
        ctx = get_step_context()
        run_name = re.sub(
            r"[^a-z0-9-]", "-", ctx.pipeline_run.name.strip().lower()
        )[:128].rstrip("-")
    except Exception:
        run_name = None

    experiment_name = (
        config.get("experiment_name", "headway-tft").lower().replace("_", "-")
    )
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
            logger.info("Logged metrics to experiment run: %s", run_name)
    except Exception as exc:
        logger.warning("Vertex AI Experiments logging failed: %s", exc)

    # -- 4. Prediction plot -------------------------------------------------
    logger.info("Building prediction plot ...")
    time_anchor = pd.Timestamp(time_anchor_iso)
    df = _build_eval_dataframe(predictions, x, test_dataset, time_anchor)
    fig = build_prediction_figure(df)
    rush_html = HTMLString(fig.to_html(full_html=True, include_plotlyjs="cdn"))

    # -- 5. TFT interpretation ----------------------------------------------
    logger.info("Generating TFT interpretation ...")
    try:
        interp_html = HTMLString(build_interpretation_html(model, raw_prediction.output))
    except Exception as exc:
        logger.warning("Interpretation failed: %s", exc)
        interp_html = HTMLString(
            f"<html><body><p>Interpretation unavailable: {exc}</p></body></html>"
        )

    return mae, smape, rush_html, interp_html

# Evaluation Step — Finalization Plan

**Date:** 2026-02-12  
**Status:** Pre-implementation  
**Scope:** 3 remaining issues in `mlops_pipeline/src/steps/evaluate_model.py`  
**Target file:** `mlops_pipeline/src/steps/evaluate_model.py` (+ test update)

---

## Task 1: Fix Rush Hour Window Targeting

### Problem

`plot_rush_hour()` picks a display window centred on the **median `time_idx`** of each group in the test set. This has no awareness of actual time-of-day — it may land on 2 AM or noon. The method is called "Rush Hour" but never checks for rush hour.

### Current code (lines 191–194)

```python
if start_idx_window is None:
    mid = group_df["time_idx"].median()
    t_start = mid - (window_size / 2)
```

### Fix

Replace the naïve median window with timestamp-aware rush hour detection:

1. **Define rush hour bands** as constants:
   ```
   AM_RUSH = (7, 10)   # 07:00–10:00
   PM_RUSH = (17, 20)   # 17:00–20:00
   ```

2. **When `time_anchor_iso` is available** (timestamps reconstructed):
   - For each group, compute the hour-of-day from the `timestamp` column.
   - Filter to rows where `hour in range(7,10)` or `hour in range(17,20)`.
   - **Prefer AM rush** (more data, commuters more consistent); fall back to PM; fall back to median if neither has data.
   - Set `t_start` to the `time_idx` of the first rush-hour row and `t_end = t_start + window_size`.

3. **When timestamps are not available** (no anchor): keep median fallback as-is.

4. **Add a subtitle annotation** to each subplot indicating which rush period was selected:
   `"AM Rush (07:00–10:00)"` or `"PM Rush (17:00–20:00)"` or `"(no rush hour data — median window)"`.

### Why this works in production evaluation

Every production retraining run calls `clean_dataset()` which computes a fresh `time_anchor_iso = min(arrival_time_dt).isoformat()` from the current data pull. The eval step receives this anchor, reconstructs real timestamps, and can reliably find 7–10 AM / 5–8 PM windows regardless of when the training data was collected. No hardcoded dates or offsets — the rush-hour detection is purely based on hour-of-day.

### Test update

Update `mlops_pipeline/tests/test_rush_hour_viz.py` to:
- Use realistic group_ids (`A_South`, `C_N`, `E_South`)
- Generate synthetic data spanning 06:00–11:00 so AM rush is present
- Assert that the x-axis window falls within the 07:00–10:00 band

---

## Task 2: Ensure test_mae and test_smape are visible in Vertex AI Experiments

### Problem

The eval step logs metrics via:
```python
context = get_step_context()
run_name = context.pipeline_run.name          # raw: "headway_training_pipeline-2026_02_12-..."
aiplatform.start_run(run_name, resume=True)    # tries to resume
```

But ZenML's Vertex tracker **sanitizes** the run name before creating it:
```python
re.sub(r"[^a-z0-9-]", "-", name.strip().lower())[:128].rstrip("-")
```

This converts underscores → hyphens. The eval step passes the **raw** name (with underscores), which doesn't match the sanitized run Vertex AI actually created. Result: `start_run` either silently creates a *new orphan run* or throws a NotFound error caught by `except Exception`, and the metrics disappear.

### Fix

Apply the same sanitization before calling `start_run`:

```python
import re

context = get_step_context()
raw_name = context.pipeline_run.name
run_name = re.sub(r"[^a-z0-9-]", "-", raw_name.strip().lower())[:128].rstrip("-")

aiplatform.start_run(run_name, resume=True)
aiplatform.log_metrics({"test_mae": mae, "test_smape": smape})
aiplatform.end_run()
```

### Verification

After the next pipeline run, verify in the GCP console:
`Vertex AI → Experiments → {experiment_name} → {run_name}` should show:
- `best_val_loss` (from train step)
- `test_mae` and `test_smape` (from eval step)

All on the **same run row**.

---

## Task 3: Make prediction plot a shareable artifact

### Problem

Currently the HTML plot is uploaded to a private GCS path:
```
gs://mlops-artifacts-realtime-headway-prediction/tensorboard_logs/evaluation/rush_hour_performance.html
```

Accessing it requires `gcloud` auth. Stakeholders without GCP access (product, ops) cannot view it.

### Fix: Two-pronged approach

#### 3a. Generate a GCS Signed URL (short-lived, secure)

After uploading the HTML to GCS, generate a **V4 signed URL** valid for 7 days:

```python
blob = bucket.blob(blob_path)
blob.upload_from_filename(local_path)
signed_url = blob.generate_signed_url(
    version="v4",
    expiration=datetime.timedelta(days=7),
    method="GET",
)
logger.info("Shareable plot URL (valid 7 days): %s", signed_url)
```

Log the signed URL as:
1. A Vertex AI Experiment metric/param so it appears in the Experiments UI
2. A ZenML artifact metadata field via `get_step_context().add_output_metadata()`

#### 3b. Log the plot as a Vertex AI Experiment artifact

Use the Vertex AI SDK to associate the HTML as an experiment artifact:

```python
aiplatform.log_classification_metrics(...)  # Not applicable — use log_metrics + artifact ref
```

Since Vertex AI Experiments doesn't natively render HTML, the signed URL logged as a parameter is the most practical approach:

```python
aiplatform.log_params({"evaluation_plot_url": signed_url})
```

This makes the link clickable directly from the Experiments comparison table in the GCP console — any stakeholder with the URL can open the interactive plot in their browser, no GCP credentials required.

#### Signing credential note

`generate_signed_url()` requires a service account key or runs under Workload Identity on Vertex AI Pipelines. On Vertex AI, the default service account can sign blobs if the `iam.serviceAccountTokenCreator` role is granted. If this fails, fall back to logging the `gs://` URI with a note to use `gcloud storage sign-url`.

### Test update

Add a unit test that mocks `google.cloud.storage` and verifies `generate_signed_url` is called with `expiration=timedelta(days=7)`.

---

## Implementation Order

| Step | Task | Risk | Est. effort |
|------|------|------|-------------|
| 1    | Run name sanitization (Task 2) | Low — isolated 3-line fix | 5 min |
| 2    | Rush hour window targeting (Task 1) | Medium — logic change in plot method | 20 min |
| 3    | Signed URL + experiment param logging (Task 3) | Medium — GCS signing perms | 15 min |
| 4    | Update unit test | Low | 10 min |
| 5    | Local verification | — | 5 min |
| 6    | Pipeline run + Vertex AI console verification | — | ~30 min (pipeline runtime) |

**Total implementation time: ~55 min + pipeline run**

---

## Post-evaluation: Transition to Deployment

Once all 3 tasks pass in a pipeline run, Phase 1 (Training Validation) in `docs/finalize_prediction_service.md` is complete and we move to Phase 2 (Model Export + Registry).

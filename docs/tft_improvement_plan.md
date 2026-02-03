# Headway TFT Model Audit and Improvement Plan

## 1. Audit Findings

### Data Representation & Feature Engineering
*   **Broken Time Index (`time_idx`):**
    *   **Issue:** `time_idx` is generated before `data.dropna(subset=['service_headway'])` runs. This creates gaps in the integer sequence (e.g., 1, 2, [gap], 4), specifically interrupting the sequential logic required by the Transformer.
    *   **Impact:** The model interprets gaps as missing data or distinct sequences, reducing data efficiency.
    *   **Fix:** Re-generate `time_idx` *after* all filtering and sorting is complete.

*   **Handling "-1.0" for Express Trains:**
    *   **Issue:** Missing travel times (e.g., 23rd St for Express trains) are filled with `-1.0`.
    *   **Impact:** Neural networks treat inputs as continuous magnitudes. `-1.0` is mathematically close to `0.0`, confusing the model.
    *   **Fix:**
        1.  Create a boolean feature `stops_at_23rd` (1 if stop exists, 0 if express).
        2.  Fill the missing `travel_time_23rd` with the mean of valid trips.

*   **Missing Interaction Features (A/C/E Problem):**
    *   **Issue:** The TFT treats each `group_id` as isolated. The A train model cannot see C/E train congestion.
    *   **Fix:** Add "Context Features" (e.g., `num_local_trains_in_last_15min`) to the A-train rows.

### Model Architecture & Training Setup
*   **Gradient Clipping (`gradient_clip_val=0.1`):**
    *   **Issue:** Extremely low. Caps parameter updates too tightly, preventing the model from moving weights effectively.
    *   **Fix:** Increase to `1.0` or `0.5`.

*   **Model Capacity (`hidden_size=32`):**
    *   **Issue:** Insufficient for capturing complex interactions for ~75k rows.
    *   **Fix:** Increase `hidden_size` to `64`.

*   **Encoder Length (`max_encoder_length=12`):**
    *   **Issue:** Only covers very recent history (1-2 hours).
    *   **Fix:** Increase to `24` to capture longer-term cyclical delays.

---

## 2. Improvement Plan: Next Iteration

### A. Data Pipeline Repairs
1.  **Fix `time_idx` Continuity:** Move the `time_idx` generation to the **very end** of the pipeline, immediately after the final `dropna` and sort operations.
2.  **Express Train Representation:** Replace `-1.0` in travel times with the mean and add `stops_at_23rd` boolean flag.
3.  **Cross-Line Context:** (If possible in this sprint) Feature engineer `time_since_last_local_train` for Express lines.

### B. Model Configuration Changes
1.  **Relax Gradient Clipping:** Change `gradient_clip_val` from `0.1` to `1.0`.
2.  **Increase Capacity:** Change `hidden_size` from `32` to `64`.
3.  **Extend History:** Change `max_encoder_length` from `12` to `24`.

### C. Evaluation & Visualization
1.  **Plotting Mode:** Use `mode="quantiles"` (where supported) or manual unscaling for plots to ensure the "Operator View" plots are readable minutes.
2.  **Expectation:** By fixing the `time_idx` gaps and allowing larger gradients, the model should shift from overfitting the `empirical_median` (schedule) to actually using the `preceding_train_gap` (dynamic delay) signal.

# Model Optimization Plan: Bridging the Gap to SOTA

**Objective:** Reduce prediction error (RMSE) from current baseline (~170s) to match the benchmarks in the research abstract (~55s).

## 1. The Performance Gap
| Metric | Current Baseline | Target (Research Paper) | Gap |
|--------|------------------|-------------------------|-----|
| **RMSE** | ~170 seconds | ~55 seconds | **3.0x** |
| **MAE** | ~80 seconds | N/A | - |
| **Data Source** | Real MTA (Noisy) | SimMetro (Perfect Physics) | High |

## 2. Root Cause Analysis & Strategy
The research paper achieved these results using simulated data (SimMetro). Real-world MTA data introduces noise (phantom trains, GPS drift) and lacks the "perfect physics" of a simulation. However, a 3x gap suggests fundamental architectural or feature-level deficiencies in our current setup.

### A. Feature Enrichment (High Priority)
**Problem:** Our model currently only sees "Relative History" (the past 30 mins). It lacks **Global Temporal Context**.
*   *Example:* A 10-minute headway at 3:00 AM is normal. A 10-minute headway at 8:00 AM is a massive delay. The model cannot distinguish these scenarios without knowing the "Time of Day".
*   **Solution:** Inject two new inputs into the dense layers of the model:
    1.  **Time of Day Embedding:** (Sin/Cos of Minute-of-Day).
    2.  **Day of Week Embedding:** (One-hot or Integer).
    *   *Hypothesis:* This allows the model to learn the "Baseline Schedule" implicitly.

### B. Model Capacity Upgrade
**Problem:** We may be underfitting.
*   **Current:** We ran the smoke test with `FILTERS=16` (Debug Mode) or `64` (Config default). The paper likely uses deeper/wider networks.
*   **Solution:** 
    *   Increase `FILTERS` to 128.
    *   Add Residual Connections (ResNet blocks) inside the ConvLSTM decoder to allow deeper gradient flow.

### C. Data Cleaning & Alignment
**Problem:** "Impossible" headways in real data screw up MSE training.
*   **Issue:** Real data often has "Ghost Trains" (0-second headways) or "Missing Packets" (30-minute gaps that aren't real).
*   **Impact:** MSE squares these errors, causing the model to panic and "smooth out" predictions to be safe, leading to mediocre RMSE.
*   **Solution:** 
    *   **Clip Headways:** Hard cap inputs and targets at `[0.5 min, 30 min]`.
    *   **Filter Outliers:** Drop samples where the "True Target" is physically impossible (e.g., changes by >500% in 1 minute).

## 3. Execution Roadmap

### Step 1: Baseline Verification (Today)
*   Ensure we are training with `FILTERS=64` (verify `DEBUG_MODE=False` is actually using the robust config).
*   Switch Loss Function to `Huber Loss` (Robust Regression) instead of MSE. This reduces the impact of data noise on the gradients.

### Step 2: Architecture Update (Day 2)
*   Modify `SubwayDataGenerator` to output `time_of_day`.
*   Modify `st_covnet.py` to accept the auxiliary input and concatenate it before the final `Dense` layers.

### Step 3: Hyperparameter Sweep (Day 3)
*   Run 4 experiments:
    1.  Filters: 64 vs 128
    2.  Lookback: 30 vs 60 minutes
    3.  Loss: MSE vs Huber
    4.  Learning Rate: 1e-3 vs 1e-4

## 4. Acceptance Criteria
We move to the Production Pipeline (Beam/Dataflow) only when:
*   **Validation MAE** < 60 seconds (1 minute average error).
*   **Validation RMSE** < 90 seconds (1.5 minutes).
*   *Note:* Matching the SimMetro 55s RMSE might be impossible with real data, but sub-90s is realistic.

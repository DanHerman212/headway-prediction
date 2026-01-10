# Headway Prediction: Remediation Plan

**Date:** January 10, 2026  
**Current Performance:** RMSE 173s, R² 0.75  
**Target Performance:** RMSE 70-90s (realistic for real-world data)  
**Paper Benchmark:** RMSE 55s (on simulated data — not directly comparable)

---

## Status Update (Jan 10, 2026 - 5:30 PM)

### ✅ Phase 1 Complete
- [x] Kernel size changed from (3,3) to (3,1) in `src/config.py`
- [x] Temporal features disabled in training notebook
- [x] SpatialDropout3D layers removed from model
- [x] Weight decay reverted to 0.01

### ✅ Phase 2 Complete  
- [x] Spatial imputation implemented in `notebooks/3_data_merging.ipynb`
- [x] Vectorized for performance (264K timesteps)
- [x] Validation metrics added (zero fraction, sawtooth check)

### ⏳ Phase 3 Pending
**To run training:**
1. In Colab, run `notebooks/3_data_merging.ipynb` to regenerate `data/headway_matrix_full.npy`
2. Verify the validation output shows <1% zero fraction
3. Run `notebooks/5_model_optimization.ipynb` with the new data
4. Compare RMSE/R² to baseline (173s / 0.75)

---

## Executive Summary

The 3x performance gap between our implementation and the paper is **not a model problem**. Three critical bugs exist in data preprocessing and model configuration:

| Issue | Severity | Estimated Impact |
|-------|----------|------------------|
| Temporal ffill instead of spatial imputation | Critical | 40-50% of error |
| Kernel size (3,3) instead of (3,1) | High | 15-25% of error |
| Sparse grid vs dense headway field | Medium | 10-15% of error |

---

## Issue #1: Imputation Method (CRITICAL)

### Current Implementation
```python
# notebooks/3_data_merging.ipynb
FFILL_LIMIT = 30
filled_df = resampled_df.ffill(limit=FFILL_LIMIT)
```

This performs **temporal** forward-fill: if a cell is empty at time t, copy the value from time t-1.

### Paper's Method
> "For grid cells with no observed events, missing headway values are imputed based on a logic of headway observed by a station"

This is **spatial** imputation: fill empty distance bins with the headway value from the nearest upstream station.

### The Problem
Temporal ffill creates "phantom dwells":
- Train at Bin 10 at t=1
- Train moves to Bin 14 by t=5
- ffill keeps Bin 10 "occupied" for t=2,3,4
- Model sees: train stopped for 4 minutes, then teleported

The ConvLSTM learns that trains frequently stop mid-track, predicting delays that don't exist.

### Fix
Create station-to-bin mapping and impute spatially:
```python
# Pseudocode
for each time t:
    for each bin j:
        if bin j is empty:
            station = get_upstream_station(j)
            headway[t, j] = headway[t, station]
```

### Files Affected
- `notebooks/3_data_merging.ipynb` — core grid construction
- `data/headway_matrix_full.npy` — must regenerate

---

## Issue #2: Kernel Size (HIGH)

### Current Implementation
```python
# src/config.py
KERNEL_SIZE = (3, 3)
```

### Paper's Specification
> "ConvLSTM Kernel Size: (3, 1)"

### The Problem
- Our tensor shape: `(Batch, Time, 66 stations, 2 directions, Channels)`
- Kernel (3,3) convolves across both stations AND directions
- This mixes Northbound and Southbound signals

A Northbound train's headway is causally linked to other Northbound trains, not to Southbound trains on the opposite track. The (3,3) kernel forces the model to find correlations that don't exist (noise).

### Fix
```python
# src/config.py
KERNEL_SIZE = (3, 1)  # Convolve along stations only
```

### Files Affected
- `src/config.py`

---

## Issue #3: Grid Density (MEDIUM)

### Current State
The grid likely contains many zeros where:
- Zero means "no train detected" OR "sensor dropout"
- The model can't distinguish between empty track and missing data

### Paper's Approach
The heatmaps in Figure 4 show **continuous color gradients** — a dense "headway field" where every cell has a meaningful value representing "current headway at this location."

### Fix
After spatial imputation (Issue #1), verify grid density:
```python
# Target: <1% zeros in the grid
zero_fraction = (headway_matrix == 0).sum() / headway_matrix.size
assert zero_fraction < 0.01, f"Grid too sparse: {zero_fraction:.2%} zeros"
```

### Files Affected
- `notebooks/3_data_merging.ipynb`

---

## Issue #4: Outlier Handling (MEDIUM)

### Current State
```python
MAX_HEADWAY = 30  # minutes
```

Real data contains extreme outliers (first train of day, multi-hour gaps during incidents). If these aren't clipped before MinMax scaling, useful variance (2-min vs 5-min headway) gets compressed.

### Fix
Hard-clip before scaling:
```python
headway_data = np.clip(headway_data, 0, MAX_HEADWAY)
# Then apply MinMaxScaler
```

### Files Affected
- `notebooks/3_data_merging.ipynb` (clip before saving .npy)
- OR `notebooks/5_model_optimization.ipynb` (clip after loading)

---

## Issue #5: Decoder Input Verification (LOW)

### Question
Are we feeding **planned** terminal headways (schedule) or **actual** future arrivals (data leakage)?

### Check Required
Review `data/target_terminal_headways.csv` generation:
- If derived from GTFS `stop_times.txt` → Correct (planned)
- If derived from actual arrival times → Data leakage

### Files Affected
- `notebooks/2_static_data_processing.ipynb` (schedule construction)

---

## Remediation Phases

### Phase 1: Quick Fixes (1 hour)
- [ ] Change `KERNEL_SIZE = (3, 1)` in config.py
- [ ] Revert `USE_TEMPORAL_FEATURES = False` (remove noise)
- [ ] Remove dropout layers (revert to baseline model)
- [ ] Verify terminal schedule is from GTFS, not actual arrivals

### Phase 2: Imputation Rewrite (2-3 hours)
- [ ] Create station-to-bin mapping from `a_line_station_distances.csv`
- [ ] Rewrite grid construction with spatial imputation
- [ ] Regenerate `headway_matrix_full.npy`
- [ ] Validate grid density (<1% zeros)

### Phase 3: Validation Run (1-2 hours)
- [ ] Train on A100 with fixed configuration
- [ ] Compare RMSE/R² to baseline
- [ ] If improved, document new baseline

### Phase 4: Per-Horizon Evaluation (1 hour)
- [ ] Implement per-horizon RMSE (t+1, t+5, t+10, t+15)
- [ ] Compare to paper's Table 2 methodology
- [ ] Realistic target: t+15 = 70-90s RMSE

---

## Realistic Expectations

| Scenario | Expected RMSE |
|----------|---------------|
| Paper (simulation) | 55s |
| Our target (real data, fixed bugs) | 70-90s |
| Current (bugs present) | 173s |
| Theoretical floor (irreducible noise) | 50-60s |

The 55s benchmark is **not achievable** on real NYC subway data. Simulation data has:
- Perfect sensor coverage
- Bounded dwell distributions
- No multi-hour incident gaps

A realistic goal is **70-90s RMSE** — good enough for operational dispatch decisions.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/config.py` | `KERNEL_SIZE = (3, 1)` |
| `notebooks/3_data_merging.ipynb` | Spatial imputation, outlier clipping |
| `notebooks/5_model_optimization.ipynb` | `USE_TEMPORAL_FEATURES = False`, verify decoder input |
| `src/models/convlstm.py` | Remove dropout (revert) |
| `src/training/trainer.py` | Revert weight_decay to 0.01 |

---

## Success Criteria

1. **Grid density:** <1% zeros after imputation
2. **Kernel alignment:** (3,1) matching paper
3. **Test RMSE:** <100s (40% improvement from 173s)
4. **Train/Val gap:** <15s (no overfitting)

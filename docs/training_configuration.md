# ConvLSTM Training Configuration

## Current Configuration (January 10, 2026)

### Optimizer

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | **AdamW** | Decoupled weight decay |
| Learning Rate | CosineDecay with 5-epoch warmup | |
| Initial LR | 1e-6 (warmup start) | |
| Peak LR | 1e-3 (after warmup) | |
| Final LR | 1e-5 (1% of peak) | |
| β₁ | 0.9 | Default |
| β₂ | **0.95** | Faster curvature tracking (was 0.999) |
| ε | **1e-6** | bfloat16 safety buffer (was 1e-7) |
| weight_decay | **0.01** | AdamW decoupled regularization |
| Gradient clipping | clipnorm=1.0 | Per-layer clipping |

### Architecture

| Component | Value | Notes |
|-----------|-------|-------|
| Model | ConvLSTM Encoder-Decoder | |
| Encoder layers | 2 × ConvLSTM2D | |
| Decoder layers | 2 × ConvLSTM2D | |
| Filters | 32 | |
| Kernel size | (3, 3) | |
| Normalization | **GroupNormalization(32)** | Was BatchNorm |
| ConvLSTM unroll | True | |
| Output activation | ReLU | |

### Training

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 256 | Optimized for A100 |
| Epochs | 100 | |
| Precision | mixed_bfloat16 | |
| XLA | jit_compile=True | |
| Early stopping | patience=20 | |
| ReduceLROnPlateau | **Removed** | Conflicts with CosineDecay |

### Data

| Parameter | Value |
|-----------|-------|
| Input shape | (30, 66, 2) — 30 min lookback, 66 stations, 2 directions |
| Output shape | (15, 66, 2) — 15 min forecast |
| Normalization | MinMaxScaler to [0, 1] (max=30 min) |
| Schedule input | (15, 2) — terminal scheduled headways |

---

## Configuration History

### Run 3: AdamW + GroupNorm (Current)

**Changes from Run 2:**
- Adam → **AdamW** with weight_decay=0.01
- β₂: 0.999 → **0.95**
- ε: 1e-7 → **1e-6**
- BatchNorm → **GroupNorm(32)**
- Removed **ReduceLROnPlateau** (conflicted with CosineDecay)

**Results (epoch 26 before callback crash):**
- Train R²: 0.80, RMSE: 167s
- Val R²: 0.75, RMSE: 170s
- **Stable** — no numerical instability
- Crashed due to ReduceLROnPlateau conflict (now fixed)

### Run 2: CosineDecay + Warmup + BatchNorm

**Changes from Run 1:**
- Added 5-epoch LR warmup (1e-6 → 1e-3)
- CosineDecay schedule
- clipnorm=1.0

**Results:**
- Survived epoch 8 (original crash point)
- **Crashed at epoch 41** — Edge of Stability spike
- Val R²: ~0.73, RMSE: ~175s before crash

### Run 1: Baseline (Original)

**Configuration:**
- Adam with default ε=1e-7, β₂=0.999
- No warmup, constant LR=1e-3
- BatchNormalization
- mixed_bfloat16, batch=256

**Results:**
- **Crashed at epoch 8** — NaN explosion
- Fast initial learning but unstable

---

## Root Cause Analysis

### Epoch 8 Crash (Run 1)
- **Cause:** No warmup → optimizer hit high-curvature region before moment estimates stabilized
- **Fix:** 5-epoch LR warmup

### Epoch 41 Crash (Run 2)
Three converging factors:
1. **Adam ε underflow:** ε=1e-7 too small for bfloat16 when gradients vanish
2. **BatchNorm instability:** Batch statistics drift across RNN time steps
3. **Edge of Stability:** β₂=0.999 has 1000-step memory, underestimates sharpness

**Fix:** AdamW (β₂=0.95, ε=1e-6) + GroupNorm

### Epoch 26 Crash (Run 3)
- **Cause:** ReduceLROnPlateau tried to modify LR, but CosineDecay schedule is immutable
- **Fix:** Remove ReduceLROnPlateau callback

---

## Files Modified

- `src/training/trainer.py`: AdamW optimizer, removed ReduceLROnPlateau
- `src/models/convlstm.py`: GroupNormalization(32) replacing BatchNormalization

---

## Reference: Paper Configuration

Based on: Usama & Koutsopoulos (2025) arXiv:2510.03121

| Parameter | Paper Value |
|-----------|-------------|
| Lookback | 30 min |
| Forecast | 15 min |
| Batch size | 32 |
| Epochs | 100 |
| Early stopping | 50 epochs |
| Filters | 32 |
| Learning rate | Adam default (1e-3) |

# Training Speedup Plan

**Date:** January 7, 2026  
**Goal:** Train a model in ≤1 hour on Vertex AI with A100 GPU  
**Current State:** 12 minutes per epoch (unacceptable)  

---

## Root Cause Analysis

### Current Performance Breakdown

| Metric | Value |
|--------|-------|
| Training samples | 158,506 |
| Batch size | 32 |
| Batches per epoch | ~4,953 |
| Time per epoch | 12 minutes |
| Time per batch | ~0.145 seconds |
| Model architecture | 3× ConvLSTM2D layers |

### Why ConvLSTM Is Slow

ConvLSTM2D processes sequences **step-by-step** (30 timesteps). Each step:
1. Applies 2D convolution (kernel 3×3) across 66×2 spatial grid
2. Updates LSTM gates (4× weight matrices per layer)
3. Cannot parallelize across timesteps (sequential dependency)

With 3 ConvLSTM layers × 30 timesteps = **90 sequential convolution operations per sample**.

### Additional Issues Found

1. **Data saved as float64** - The `.npy` files are float64 (0.28 GB), wasting memory bandwidth
2. **No mixed precision** - A100 has TensorCores that provide 2× speedup with float16
3. **Batch size too small** - A100 has 40GB memory; batch_size=32 underutilizes it
4. **Lambda layers** - `tf.tile` in Lambda causes graph optimization issues

---

## Solution: Two-Track Approach

### Track 1: Quick Wins (Keep ConvLSTM Architecture)

These changes preserve the paper-faithful architecture while maximizing speed:

| Change | Expected Speedup | Implementation |
|--------|------------------|----------------|
| Increase batch_size to 256 | ~3× | Config change |
| Enable mixed precision (float16) | ~2× | 2 lines of code |
| Use compiled model (`jit_compile=True`) | ~1.3× | 1 line of code |
| Remove Lambda layers (use Keras ops) | ~1.1× | Refactor model |

**Combined expected speedup: ~8×**  
**Projected time per epoch: ~1.5 minutes**  
**30 epochs in ~45 minutes** ✓

### Track 2: Simplified Architecture (Faster Baseline)

If Track 1 is insufficient, replace ConvLSTM with a simpler architecture:

**Option A: Conv3D Only (No Recurrence)**
```
Input(30, 66, 2, 1) → Conv3D(32) → Conv3D(32) → Conv3D(15) → Output(15, 66, 2, 1)
```
- No sequential processing
- Fully parallelizable
- ~10× faster than ConvLSTM

**Option B: Temporal Fusion Transformer (TFT) Style**
```
Input → Dense embedding → Multi-head attention → Dense → Output
```
- Attention parallelizes across time
- Better for longer sequences
- Industry standard for time series

---

## Implementation Plan for Tomorrow

### Phase 1: Quick Wins (30 minutes)

#### 1.1 Enable Mixed Precision

Add to `pipeline.py` training component (before model building):

```python
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

#### 1.2 Increase Batch Size

Update `config.py`:

```python
BATCH_SIZE: int = 256  # Was 32
```

#### 1.3 Enable XLA Compilation

Update `trainer.py` compile_model():

```python
self.model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=[rmse_seconds, r_squared],
    jit_compile=True  # Add this
)
```

#### 1.4 Refactor Lambda Layers

Replace custom Lambda layers in `baseline_convlstm.py` with Keras built-in operations:

```python
# Instead of Lambda with tf.tile:
state_repeated = layers.RepeatVector(forecast)(...)  # Use native layer

# Instead of Lambda broadcast:
# Reshape and use broadcasting in Concatenate
```

### Phase 2: Validate Locally (15 minutes)

Run smoke test with reduced data:

```bash
python -m src.experiments.smoke_test
```

Verify:
- Model compiles without errors
- Training runs for 2 epochs
- Metrics are logged correctly

### Phase 3: Deploy and Monitor (15 minutes)

```bash
./run_training.sh quick
```

Monitor first 3 epochs to verify:
- Time per epoch < 2 minutes
- Loss is decreasing
- No OOM errors

---

## Code Changes Required

### File: `src/config.py`

```python
# Change:
BATCH_SIZE: int = 256  # Was 32
```

### File: `src/training/trainer.py`

```python
# In compile_model(), add jit_compile:
self.model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=[rmse_seconds, r_squared],
    jit_compile=True
)
```

### File: `src/experiments/pipeline.py`

Add at the start of `training_component`:

```python
# Enable mixed precision for A100 TensorCores
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### File: `src/models/baseline_convlstm.py`

Replace Lambda layers with native Keras operations (detailed refactor needed).

---

## Fallback Plan

If Track 1 fails to achieve <2 min/epoch:

1. **Reduce training data to 20%** - Quick sanity check (10K samples)
2. **Implement Conv3D baseline** - Simple architecture file already scaffolded
3. **Use pretrained weights** - Start from checkpoint instead of random init

---

## Validation Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time per epoch | < 2 minutes | Vertex AI logs |
| Training time (30 epochs) | < 1 hour | Total pipeline time |
| Final val_loss | < 0.01 | TensorBoard |
| RMSE (seconds) | < 90 seconds | Custom metric |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OOM with batch_size=256 | Medium | Start with 128, increase if stable |
| Mixed precision NaN | Low | Keep loss scaling enabled (default) |
| XLA compilation failure | Low | Disable jit_compile if errors |
| Vertex AI quota issues | Low | Already running successfully |

---

## Tomorrow's Checklist

- [ ] Apply quick win changes (config, trainer, pipeline)
- [ ] Test locally with smoke_test
- [ ] Submit to Vertex AI
- [ ] Monitor first 3 epochs
- [ ] Verify Vertex AI Experiments logging
- [ ] If >2 min/epoch, implement Conv3D fallback
- [ ] Document final results

---

## References

- [TensorFlow Mixed Precision Guide](https://www.tensorflow.org/guide/mixed_precision)
- [XLA JIT Compilation](https://www.tensorflow.org/xla)
- [A100 Performance Optimization](https://developer.nvidia.com/blog/optimizing-tensorflow-performance-on-a100/)
- Paper: Usama & Koutsopoulos (2025) arXiv:2510.03121

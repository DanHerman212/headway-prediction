# Graph WaveNet Implementation Plan

## Summary of Understanding

### What You Have

**1. Rich Research Document**: The [graph_wavenet_architecture.md](graph_wavenet_architecture.md) covers:
- **Domain Context**: A/C/E line interoperability, delay propagation, shared infrastructure
- **Architecture Selection Rationale**: Graph WaveNet chosen for non-Euclidean data, dilated TCN for long-range temporal patterns, and the **Adaptive Adjacency Matrix** to learn latent operational dependencies
- **Mathematical Foundations**: Diffusion convolution, gated TCN, spectral vs spatial approaches
- **Data Pipeline**: GTFS-RT ingestion, ghost train filtering, tensor construction
- **Deployment Strategy**: Kafka → Redis → TF Serving → FastAPI

**2. Existing ConvLSTM Implementation**: A working ConvLSTM encoder-decoder that:
- Takes `(Batch, 30, 66, 2)` headway history
- Takes `(Batch, 15, 2)` schedule input from 2 terminals
- Outputs 15-minute forecasts

**3. Prepared Data Assets**:
- `headway_matrix_topology.npy` — Headway observations
- `schedule_matrix_4terminal_planned.npy` — 4-terminal schedule inputs
- `topology_grid.csv` — 75 bins (trunk + Lefferts + Far Rockaway branches)
- `temporal_features.npy` — Time encodings

---

## Key Architectural Shift: ConvLSTM → Graph WaveNet

| Aspect | Current (ConvLSTM) | Target (Graph WaveNet) |
|--------|-------------------|------------------------|
| **Spatial** | 2D grid (stations × directions) | Graph nodes (station-direction-line tuples) |
| **Adjacency** | Implicit (CNN kernel) | Explicit + **Adaptive** (learned) |
| **Temporal** | LSTM (sequential) | Dilated TCN (parallel, exponential receptive field) |
| **Skip Connections** | Limited | Full WaveNet-style aggregation |

---

## 6-Phase Development Plan

### Phase 1: Define Prediction Task
**Goal**: Formalize inputs, outputs, and success metrics

- **Input**: 12 timesteps (1 hour @ 5-min bins) of headway + schedule + time features across ~N nodes
- **Output**: 12 timesteps (1 hour) headway predictions
- **Nodes**: Define as `(station, direction, line)` tuples or use existing 75-bin topology
- **Metrics**: Masked MAE < 2 min, MAPE, Regularity Index correlation
- **Deliverable**: `docs/prediction_task_spec.md`

### Phase 2: Create Dataset
**Goal**: Build Graph WaveNet-compatible tensors + adjacency matrices

- Reshape existing `.npy` files to `(Samples, T, N, F)` format
- Construct **physical adjacency matrix** from topology (track connections)
- Prepare node feature vectors: observed headway, scheduled headway, delay, time encoding
- Create train/val/test splits with proper temporal ordering
- **Deliverable**: `src/data/graph_dataset.py`, `data/adjacency_matrix.npy`

### Phase 3: Develop Model Architecture
**Goal**: Implement Graph WaveNet in TensorFlow/Keras

- `AdaptiveAdjacency` layer (learnable E₁E₂ᵀ)
- `DiffusionConv` layer (K-hop graph convolution)
- `GatedTCN` layer (dilated causal convolutions with GLU)
- `STBlock` (Spatio-Temporal block combining TCN + GCN)
- Full `GraphWaveNet` model with skip connections
- **Deliverable**: `src/models/graph_wavenet.py`

### Phase 4: Train to Overfit
**Goal**: Verify model capacity by memorizing a small subset

- Train on ~1 week of data with no regularization
- Target: Training MAE → near 0
- Confirm forward pass, gradients flow, loss decreases
- Debug any shape mismatches or NaN issues
- **Deliverable**: Overfitting training script, validation that architecture works

### Phase 5: Regularize for Robust Fit
**Goal**: Generalize to unseen data

- Add dropout (0.3), batch normalization
- Implement learning rate scheduling (ReduceLROnPlateau)
- Early stopping on validation loss
- Curriculum learning: start with peak hours → add off-peak
- Hyperparameter tuning: blocks, channels, diffusion steps
- Compare against ConvLSTM baseline and HA/VAR
- **Deliverable**: Trained model, evaluation report, benchmark comparisons

### Phase 6: Deploy to Production
**Goal**: Real-time inference pipeline

- Export model as SavedModel
- TensorFlow Serving container
- Redis state manager for sliding window
- FastAPI gateway
- Online retraining loop (weekly fine-tuning)
- **Deliverable**: Dockerized inference service, deployment manifests

---

## Discussion Points

1. **Node Definition**: Should we keep the 75-bin topology or move to the ~300-400 node scheme (station × direction × line) from the research document?

2. **Adjacency Matrix Construction**: Do we have track connectivity data, or should we derive it from `topology_grid.csv`?

3. **Data Granularity**: Current data is 1-minute bins (30 min lookback = 30 steps). The document suggests 5-minute bins (12 steps = 1 hour). Which is preferred?

4. **Scope**: Start with A-line only (current) or immediately incorporate C/E interlining?

5. **Framework**: Stick with TensorFlow/Keras (current codebase) or consider PyTorch Geometric (stronger GNN ecosystem)?

---

## Next Steps

Once these discussion points are resolved, we can begin Phase 1 implementation.

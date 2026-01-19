# ML Workflow For Realtime Headway Prediction

This document defines the end-to-end machine learning workflow for predicting subway headways on the NYC A/C/E line complex using Graph WaveNet.

---

## 1. Prediction Task

### 1.1 Task Type: Multi-Step Spatiotemporal Forecasting

We define the task as **Sequence-to-Sequence Regression**. We are not predicting a single value (like "when will the next train arrive?"), but rather a **series of future system states** over the next hour.

| Aspect | Specification |
|--------|---------------|
| **Goal** | Predict Effective Headway for every node in the A/C/E network for the next 12 time intervals |
| **Input** | Past 60 minutes of operational data |
| **Output** | Next 60 minutes of predicted headways |

### 1.2 Target Variable: Effective Headway

Since Graph WaveNet expects a continuous signal (like speed on a highway), we predict **Headway** as a continuous state variable rather than discrete train arrival timestamps.

**Definition:** For a specific time bin *t* at Station *s*, the "Effective Headway" is the time gap (in minutes) between the train that arrived in that bin and the previous train.

**Handling Sparsity:** If no train arrives during a 5-minute bin (common in late-night service), the system carries forward the last observed headway or uses an interpolating "decay" value to represent the growing gap.

**Why this matters:** This allows the model to predict:
- **Bunching** — e.g., predicting a drop in headway to 2 minutes at 42nd St in 30 minutes
- **Gaps** — e.g., predicting a spike to 20 minutes

### 1.3 Graph Definition

The A, C, and E lines are treated as a **single unified graph** to capture shared track interference.

**Node Definition:** A node is a unique **Station-Line-Direction** tuple.

*Examples:*
- `(42nd St, A, North)`
- `(42nd St, C, North)`
- `(42nd St, E, North)`

These are three distinct nodes even though they share the same physical station.

**Graph Size:** ~300-400 nodes (*N*)

### 1.4 Temporal Resolution & Horizon

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Time Bin** | 5 minutes | 30-sec bins too noisy; 15-min bins lose detail |
| **Input Sequence (T_in)** | 12 steps (60 min) | Captures delay propagation trends |
| **Output Sequence (T_out)** | 12 steps (60 min) | Provides 30-60 min lead time for dispatcher intervention |

### 1.5 Mathematical Formulation

Learn a function *f* that maps the historical graph signal to future graph signal:

```
f(X_{t-11:t}, A) → X_{t+1:t+12}
```

Where:
- **X ∈ ℝ^(T × N)** — Headway signal matrix (time steps × nodes)
- **A** — Graph adjacency matrix representing track topology (including shared A/C/E tracks)

### 1.6 Output Dimensions

Each inference produces a `(12, N)` matrix:
- **12** time steps × **~300** nodes = **~3,600 predicted headway values**

---

## 2. Dataset for Training

### 2.1 Data Sources

#### A. Historical Archive (Ground Truth)
- **File:** `nyc_subway_a_line_arrivals_2025.csv`
- **Records:** ~2.1 million arrivals
- **Date Range:** June 6, 2025 → December 6, 2025 (~6 months)
- **Fields Available:**
  - `trip_uid` — Unique trip identifier
  - `route_id` — A line (C/E to be added)
  - `direction` — N (North) or S (South)
  - `stop_id` — Station identifier
  - `stop_name` — Human-readable station name
  - `stop_lat`, `stop_lon` — Coordinates
  - `arrival_time` — Actual arrival timestamp (UTC)

#### B. Static GTFS (Schedule Skeleton)
Located in `data/static/`:

| File | Purpose |
|------|---------|
| `stop_times.txt` | Scheduled arrival times → Calculate lateness |
| `stops.txt` | Map `stop_id` → Node definition |
| `routes.txt` | Separate A, C, E services |
| `trips.txt` | Link trips to routes and service days |
| `calendar.txt` | Service patterns (weekday/weekend) |

#### C. Missing Data (Gap Analysis)

| Data | Status | Impact |
|------|--------|--------|
| **C/E Line Arrivals** | ❌ Not in archive | Need to acquire for unified A/C/E graph |
| **GTFS-RT Alerts** | ❌ Not archived | Cannot compute Alert Status feature |
| **Historical Static GTFS** | ⚠️ May not match archive dates | Verify schedule version alignment |

**Mitigation:** If Alerts unavailable, start with 3 features (Headway, Lateness, Time). Add Alerts in v2.

### 2.2 Graph Construction (Adjacency Matrix)

The adjacency matrix **A** encodes the track topology.

#### Node Definition
Each node is a **Station-Line-Direction** tuple:
```
(stop_id, route_id, direction)
```
*Example:* `(A38S, A, S)` = Fulton St, A train, Southbound

#### Adjacency Rules
1. **Sequential Connectivity:** Node *i* connects to Node *i+1* along the route
2. **Shared Tracks:** A/C/E share 8th Avenue tracks → cross-line edges
3. **Transfer Stations:** Optional weighted edges for passenger flow

#### Matrix Properties
- **Size:** `(N, N)` where N ≈ 300-400 nodes
- **Type:** Binary or distance-weighted
- **Storage:** Sparse matrix (most entries are 0)

### 2.3 Feature Engineering

Build a tensor for every **5-minute bin** for every **Node**.

#### Dynamic Features (Input X)

| # | Feature | Source | Calculation | Imputation |
|---|---------|--------|-------------|------------|
| 1 | **Observed Headway** | Archive `arrival_time` | `arrival[i] - arrival[i-1]` | Forward-fill "running headway" |
| 2 | **Schedule Deviation** | Static vs Actual | `actual_arrival - scheduled_arrival` | 0 if no scheduled train |
| 3 | **Alert Status** | GTFS-RT Alerts | Binary (0/1) | 0 (assume no alert if missing) |
| 4 | **Time Embedding (Sin)** | Timestamp | `sin(2π × minute_of_day / 1440)` | N/A |
| 5 | **Time Embedding (Cos)** | Timestamp | `cos(2π × minute_of_day / 1440)` | N/A |
| 6 | **Day of Week (Sin)** | Timestamp | `sin(2π × day_of_week / 7)` | N/A |
| 7 | **Day of Week (Cos)** | Timestamp | `cos(2π × day_of_week / 7)` | N/A |

**Note:** Features 4-7 are cyclical encodings; neural networks handle these better than raw integers.

#### Target Variable (Y)
- **Effective Headway** for each node at each future time step
- Same as Feature #1, but for the prediction horizon

### 2.4 Tensor Shapes

#### Dataset Dimensions

| Parameter | Value |
|-----------|-------|
| Time Bin | 5 minutes |
| Bins per Day | 288 |
| Archive Duration | ~6 months |
| Total Time Steps | ~52,000 |
| Nodes (A-line only) | ~100 |
| Nodes (A/C/E unified) | ~300-400 |
| Features | 4-7 (depending on Alert availability) |

#### Final Tensor Shapes

```
Input X:  (Samples, T_in, Nodes, Features)  = (S, 12, N, F)
Target Y: (Samples, T_out, Nodes)           = (S, 12, N)
Adjacency A: (Nodes, Nodes)                 = (N, N)
```

**Concrete Example (A-line only, 6 months):**
```
X: (51000, 12, 100, 4)   # ~51K samples, 12 lookback, 100 nodes, 4 features
Y: (51000, 12, 100)      # Predict 12 future headways per node
A: (100, 100)            # Adjacency matrix
```

### 2.5 Train/Validation/Test Split Strategy

**Temporal Split** (not random) — prevents data leakage:

| Split | Date Range | Purpose |
|-------|------------|---------|
| **Train** | June 6 – Oct 15, 2025 (~75%) | Learn patterns |
| **Validation** | Oct 16 – Nov 10, 2025 (~15%) | Hyperparameter tuning |
| **Test** | Nov 11 – Dec 6, 2025 (~10%) | Final evaluation |

**Rationale:**
- Test set includes Thanksgiving week (rare event stress test)
- No shuffling — model never sees "future" data during training

### 2.6 Data Pipeline Steps

1. **Ingest & Parse:** Load archive CSV, parse timestamps to datetime
2. **Bin Arrivals:** Aggregate into 5-minute bins per node
3. **Calculate Headways:** Compute inter-arrival times
4. **Impute Missing:** Forward-fill empty bins with running headway
5. **Join Schedule:** Match to GTFS static for lateness calculation
6. **Add Time Features:** Compute sin/cos embeddings
7. **Build Adjacency:** Construct graph from stop sequences
8. **Create Sequences:** Sliding window → (X, Y) pairs
9. **Normalize:** StandardScaler on features (fit on train only)
10. **Export Tensors:** Save as `.npy` or TFRecord

---

## 3. Model Selection and Training

*To be defined*

### 3.1 Model Architecture: Graph WaveNet

### 3.2 Training Strategy: Overfit First, Then Regularize

#### Phase 1: Overfit on Small Subset
- Validate model can learn the signal
- Confirm gradient flow and convergence

#### Phase 2: Scale and Regularize
- Add dropout, weight decay
- Tune hyperparameters

### 3.3 Loss Function

### 3.4 Evaluation Metrics

### 3.5 Baseline Comparisons

---

## 4. Deploy to Production

*To be defined — GCP infrastructure*

### 4.1 Inference Pipeline

### 4.2 Real-time Data Ingestion

### 4.3 Model Serving Infrastructure

### 4.4 Monitoring and Alerting

### 4.5 Retraining Strategy

---

## Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| **Effective Headway** | Time gap between consecutive train arrivals at a node |
| **Node** | Station-Line-Direction tuple |
| **Bunching** | When headways compress (trains too close together) |
| **Gap** | When headways expand (long wait between trains) |

### B. References

- Graph WaveNet paper: [arXiv:1906.00121](https://arxiv.org/abs/1906.00121)
- METR-LA / PEMS-BAY benchmark datasets

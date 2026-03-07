# Model Progression Log

**Project**: TFT Headway Prediction — A/C/E Southbound (NYC Subway)  
**Document**: Tracks model iterations, feature engineering results, and optimization decisions.

---

## Iteration 1: Baseline + Upstream Feature Engineering

**Date**: March 7, 2026  
**Training environment**: NVIDIA RTX PRO 6000 Blackwell Server Edition, bf16-mixed precision  
**Training time**: ~100 minutes (31 epochs × ~3:15/epoch)  
**Data**: 4,253,782 rows, 105 unique time series (route × stop groups)

### 1.1 Dataset

- **Source**: BigQuery `headway_prediction.clean`, routes A/C/E, direction S
- **Target**: `minutes_until_next_train` (headway in minutes, capped at 25 min)
- **Splits** (temporal, no data leakage):

| Split | Cutoff | Rows | Purpose |
|-------|--------|------|---------|
| Train | < 2025-10-29 | ~3.4M (80%) | Model fitting |
| Val | < 2025-12-23 | ~500K (12%) | Early stopping / tuning |
| Test | < 2026-02-17 | 652K (8%) | Final holdout evaluation |

### 1.2 Feature Engineering

Five groups of upstream/spatial features were engineered in `bigquery_explorer.ipynb` and added to the existing baseline features (rolling stats, cyclical encodings, empirical median).

**Baseline features** (pre-existing):
- Cyclical time encodings: hour_sin/cos, dow_sin/cos, month_sin/cos
- Binary flags: is_weekend, is_holiday
- Rolling headway stats: rolling_mean_3/5/10, rolling_std_3/5, rolling_max_10
- Empirical median headway (by route, is_weekend, hour — train period only)
- Static: route_id, stop_id, station_order

**New features by group:**

| Group | Features | Method | Correlation w/ target |
|-------|----------|--------|----------------------|
| **1: Same-Route Upstream Headway** | upstream_headway_1, upstream_headway_2, upstream_headway_3, upstream_deviation_ratio | `pd.merge_asof` backward, strict `<`, branch-aware upstream stop mapping | r = 0.90, 0.80, 0.70, 0.54 |
| **2: Upstream Travel Time** | last_train_travel_time_1, last_train_travel_time_2, travel_time_deviation_1 | Self-join on `trip_uid` to find same physical train at upstream stop | r = -0.015, -0.048, -0.015 |
| **3: Time Since Upstream Departure** | time_since_upstream_1, time_since_upstream_2 | `pd.merge_asof` backward for recency of upstream activity | r = 0.025, 0.217 |
| **4: Cross-Route Signals** | any_route_headway_upstream, any_route_time_since_upstream, cross_route_trains_last_10min | Route-agnostic merge_asof + vectorized searchsorted per stop | r = 0.84, 0.03, -0.274 |
| **5: Deviation Composites** | headway_deviation_ratio, headway_deviation_signed, deviation_trend, deviation_streak | Arithmetic composites of rolling stats and empirical median | r = 0.05–0.15 |

**Total features**: 36 (22 time-varying unknown reals, 9 time-varying known reals, 2 static categoricals, 1 static real, plus TFT-internal features)

### 1.3 Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Encoder length | 25 | ACF < 0.10 at lag 23, PACF ≈ 0 by lag 15 |
| Prediction length | 1 | Single-step headway prediction |
| Batch size | 256 | Balance of gradient noise and update frequency |
| Hidden size | 128 | Default, not yet tuned |
| Attention heads | 4 | Default, not yet tuned |
| Dropout | 0.1 | Low — train/val gap was only 0.04 |
| Learning rate | 1e-3 | Default, with reduce-on-plateau patience=3 |
| Gradient clip | 0.1 | Standard for TFT |
| Loss | QuantileLoss [0.1, 0.5, 0.9] | p10/p50/p90 predictions |
| Target normalizer | GroupNormalizer | Per-group scale/center |
| limit_train_batches | 1000 | ~256K samples/epoch (7.5% of train data) |
| limit_val_batches | 400 | Proportional cap |

### 1.4 Training Dynamics

- **31 epochs** before early stopping (patience=5)
- **val_loss trajectory**: 1.486 → 1.332 (first 9 epochs, all improving) → continued improving to ~1.28 at epoch 26 → plateau at epoch 26–31
- **val MAE**: 2.28 (epoch 1) → 2.10 (epoch 5) → 1.996 (epoch 26) → plateau
- **Train/val gap**: ~0.04 (minimal overfitting)
- **GPU utilization**: 91%

### 1.5 Test Set Results

| Metric | Value |
|--------|-------|
| **MAE** | **2.319 min** |
| RMSE | 3.477 min |
| MAPE | 42.6% (misleading — driven by small-headway divisions) |
| Samples | 652,354 |

**Quantile calibration** (nearly perfect):

| Quantile | Expected | Actual | Δ |
|----------|----------|--------|---|
| p10 | 10.0% | 11.1% | +1.1% |
| p50 | 50.0% | 50.8% | +0.8% |
| p90 | 90.0% | 89.2% | -0.8% |
| **80% PI** | **80.0%** | **78.1%** | **-1.9%** |

**MAE by headway range:**

| Headway Bin | MAE | Note |
|-------------|-----|------|
| 0–5 min | ~2.3 | Good — bulk of data |
| 5–10 min | ~1.7 | Best performance sweet spot |
| 10–15 min | ~2.1 | Solid |
| 15–20 min | ~3.6 | Harder — longer headways have more variance |
| 20–25 min | ~5.5 | Tail performance, small sample |
| 25–30 min | ~9.0 | Near HEADWAY_MAX cutoff, very few samples |

### 1.6 Attention Pattern

**U-shaped** attention across the 25-step encoder:
- High attention at lag -25 (oldest) — captures regime baseline
- Low attention at lags -10 to -15 — mid-range history less useful
- Rising attention at lag -1/-2 (newest) — captures current conditions
- Validates encoder length of 25: the model uses the full window

### 1.7 Variable Importance Analysis

**Static variables:**
| Variable | Importance |
|----------|-----------|
| stop_id | 27% |
| station_order | 22% |
| minutes_until_next_train_center | 21% |
| route_id | 15% |
| minutes_until_next_train_scale | 13% |
| encoder_length | 1% |

**Encoder (time-varying) — Top performers:**
| Variable | Importance | Group |
|----------|-----------|-------|
| upstream_headway_3 | 9.8% | 1 — Upstream Headway |
| upstream_deviation_ratio | 9.5% | 1 — Upstream Headway |
| empirical_median | 9.2% | Baseline |
| upstream_headway_2 | 8.0% | 1 — Upstream Headway |
| any_route_headway_upstream | 5.1% | 4 — Cross-Route |
| hour_sin | 5.0% | Temporal |
| hour_cos | 4.8% | Temporal |
| headway_deviation_signed | 4.7% | 5 — Deviation |
| travel_time_deviation_1 | 3.6% | 2 — Travel Time |
| last_train_travel_time_2 | 3.4% | 2 — Travel Time |
| upstream_headway_1 | 3.2% | 1 — Upstream Headway |

**Encoder — Prune candidates (< 1.5%):**
| Variable | Importance | Group | Reason to prune |
|----------|-----------|-------|----------------|
| time_since_upstream_1 | 0.5% | 3 — Time Since Upstream | Lowest importance of all features |
| is_weekend | 0.6% | Temporal | Redundant with dow_sin/cos |
| dow_sin | 0.7% | Temporal | dow_cos (2.8%) captures same signal |
| month_sin | 0.8% | Temporal | month_cos (1.0%) captures same signal |
| cross_route_trains_last_10min | 0.8% | 4 — Cross-Route | Hardest to productionize AND lowest value |
| headway_deviation_ratio | 0.9% | 5 — Deviation | Redundant with headway_deviation_signed (4.7%) |
| time_since_upstream_2 | 1.0% | 3 — Time Since Upstream | Entire Group 3 is low-value |
| rolling_std_5 | 1.2% | Baseline | Redundant with rolling_std_3 (2.0%) |
| rolling_max_10 | 1.3% | Baseline | Redundant with rolling_mean_10 (2.8%) |
| is_holiday | 1.3% | Temporal | Very sparse signal |
| last_train_travel_time_1 | 1.4% | 2 — Travel Time | Redundant with travel_time_deviation_1 (3.6%) |

**Decoder (known covariates):**
| Variable | Importance |
|----------|-----------|
| empirical_median | 19% |
| relative_time_idx | 14% |
| hour_cos | 14% |
| month_cos | 13% |
| is_weekend | 9% |
| dow_cos | 8% |
| is_holiday | 7% |
| month_sin | 6% |
| hour_sin | 5% |
| dow_sin | 5% |

### 1.8 Key Findings

1. **Group 1 (Upstream Headway) is the most valuable feature group** — 3 of the top 5 encoder features, totaling ~30.5% of encoder importance. Straightforward to productionize.

2. **upstream_headway_3 > upstream_headway_2 > upstream_headway_1** — the model values information from further upstream. Physically sensible: 3-stop-upstream headway is a leading indicator of what's propagating toward you, while 1-stop is nearly simultaneous.

3. **Group 3 (Time Since Upstream) is expendable** — combined 1.5% importance. Safe to drop entirely, eliminating 2 merge_asof lookups in production.

4. **cross_route_trains_last_10min** — the hardest feature to implement in production (windowed sorted set) and only 0.8% importance. Drop it. The simpler `any_route_headway_upstream` (5.1%) captures the cross-route signal far better.

5. **Quantile calibration is production-ready** — all quantiles within ±1.1% of nominal. The 80% prediction interval ("next train in X–Y minutes") is trustworthy.

6. **The model generalizes well** — train/val loss gap of only 0.04, and test MAE (2.319) is a reasonable ~0.3 min above val MAE (1.996), expected for a temporal shift into a later period.

7. **The `track` feature was not used** in this iteration. It's available in the data (20 unique values, heavily concentrated in A1/A3/D3/K1) and could help as a time-varying unknown categorical in the next round.

---

## Iteration 2: Pruning + New Features + HPO (Planned)

### 2.1 Feature Changes

**Drop (7 features):**
- time_since_upstream_1, time_since_upstream_2 (Group 3 — entire group)
- cross_route_trains_last_10min (Group 4 — low value, high production cost)
- headway_deviation_ratio (redundant with headway_deviation_signed)
- last_train_travel_time_1 (redundant with travel_time_deviation_1)
- rolling_max_10 (redundant with rolling_mean_10)
- rolling_std_5 (redundant with rolling_std_3)

**Add (candidate features):**
- `track` — time-varying unknown categorical, signals express/local track and reroutes
- `schedule_deviation` — actual arrival vs GTFS static scheduled time (requires GTFS static join)
- `active_trip_count` — distinct trip_uids on same route in last 30 min (computable from existing data)

### 2.2 HPO Sweep (Planned)

| Parameter | Search range | Method |
|-----------|-------------|--------|
| learning_rate | 1e-4 to 3e-3 (log) | Optuna |
| dropout | 0.05 to 0.3 | Optuna |
| hidden_size | 64, 128, 256 | Optuna |
| attention_heads | 2, 4, 8 | Optuna |
| batch_size | 256, 512 | Manual comparison |

### 2.3 Data Coverage Test (Planned)

- Run with BATCH_SIZE=512, limit_train_batches=1000 (~512K samples/epoch, 15% coverage)
- Compare val_loss curve to baseline 256-batch run
- If meaningful improvement, consider full-epoch training via DDP

---

## Production Transition Plan

### Architecture (Event-Driven, Dataflow State Only)

```
GTFS-RT Feed → Pub/Sub → Dataflow Streaming Pipeline
    ├── Parse arrival event
    ├── Update Dataflow per-key state:
    │     ├── headway_buffer[route][stop] — circular buffer, size 10
    │     ├── last_headway[route][stop] — for upstream lookups
    │     ├── trip_positions[trip_uid][stop] — for travel time features
    │     └── last_arrival[stop] — route-agnostic, for cross-route
    ├── Compute features from state (~15 features after pruning)
    ├── Run TFT inference (Vertex AI endpoint)
    └── Publish prediction → Pub/Sub → Dashboard / Mobile API
```

### Production Feature Set (After Pruning)

**Tier 1 — Stateless (trivial):**
- hour_sin/cos, dow_sin/cos, month_sin/cos
- is_weekend, is_holiday
- empirical_median (static lookup table)
- station_order (static mapping)

**Tier 2 — Simple stateful (per-group counters):**
- rolling_mean_3/5/10, rolling_std_3
- headway_deviation_signed, deviation_trend, deviation_streak

**Tier 3 — Cross-group lookups:**
- upstream_headway_1/2/3, upstream_deviation_ratio
- any_route_headway_upstream, any_route_time_since_upstream
- travel_time_deviation_1, last_train_travel_time_2

**State footprint**: ~300 bytes × 105 groups = ~31 KB total (negligible)

### Key Decision: No Redis Required

All state lives in Dataflow stateful DoFns. Benefits:
- Exactly-once per key (guaranteed by Beam)
- Zero network latency (in-process state)
- Automatic checkpointing and rebalancing
- One less service to operate

Redis only needed if a separate serving path (e.g., API endpoint outside the event-driven flow) requires feature state access.

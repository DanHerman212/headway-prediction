# Data Representation for Stacked GRU Headway Prediction

## Overview

This document outlines the data pipeline for building a Multi-Head Stacked GRU model to predict train headways at West 4th Street–Washington Square station (A32) on the A, C, and E lines.

## Target Station

| Property | Value |
|----------|-------|
| Station | West 4th Street–Washington Square |
| GTFS Stop ID | A32N (Northbound), A32S (Southbound) |
| Lines | A, C, E (IND Eighth Avenue Line) |
| Level | Upper Level |
| Platform Type | Two island platforms (4 tracks total) |

### Track Layout

| Direction | Track | Role | Action |
|-----------|-------|------|--------|
| Southbound (A32S) | A1 | Local | **Keep** |
| Southbound (A32S) | A3 | Express | Exclude |
| Northbound (A32N) | A2 | Local | **Keep** |
| Northbound (A32N) | A4 | Express | Exclude |

### Service Patterns

| Time Period | Local Tracks (A1/A2) | Express Tracks (A3/A4) |
|-------------|----------------------|------------------------|
| Daytime (06:00–23:00) | C, E | A |
| Overnight (23:00–06:00) | A, E | — |

**Key Insight:** The "Night Shift" mode moves the A train from express to local tracks when C service is suspended.

---

## Stage 1: Extract Raw Dataset

**Source:** `headway_dataset.clean`  
**Target:** `headway_dataset.ml`  
**Filter:** `stop_id IN ('A32N', 'A32S')`, `route_id IN ('A', 'C', 'E')`, `track IN ('A1', 'A2')`  
**Date Range:** Sep 13, 2025 → Jan 13, 2026 (4 months)

### Output Schema

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `route_id` | STRING | Direct | A, C, or E |
| `direction` | STRING | Direct | N or S |
| `stop_id` | STRING | Direct | A32N or A32S |
| `arrival_time_ts` | TIMESTAMP | Direct | Train arrival timestamp (UTC) |
| `stop_name` | STRING | Direct | Station name |
| `day_type` | STRING | Direct | Weekday or Weekend |
| `track` | STRING | Direct | A1 (Southbound local) or A2 (Northbound local) |
| `prev_arrival_time` | TIMESTAMP | Computed | Previous train arrival on same track |
| `headway_total_seconds` | INT | Computed | Seconds since previous train (composite) |
| `headway_minutes` | INT | Computed | Minutes component |
| `headway_seconds_remainder` | INT | Computed | Seconds component |
| `headway_display` | STRING | Computed | MM:SS format |

### Headway Logic

**Composite Headway:** Time since *any* previous train on the same track, regardless of line.

```sql
PARTITION BY stop_id, track
ORDER BY arrival_time_ts
```

**Rationale:** C and E trains share the local track. Physical signal blocking means a delayed C train blocks the following E train. The model must learn from the full composite sequence.

---

## Stage 2: Exploratory Data Analysis (EDA)

### Analysis Checklist

- [ ] **Headway Distribution**
  - Histogram of raw headways
  - Confirm log-normal / right-skewed distribution
  - Identify outlier threshold (currently 120 min)

- [ ] **A vs C vs E Comparison**
  - Mean/median headway by line
  - Service frequency by time of day
  - Composite sequence patterns

- [ ] **Directional Asymmetry**
  - A1 (Southbound) vs A2 (Northbound) distributions
  - Confirm need for separate models

- [ ] **Temporal Patterns**
  - Headway by hour (peak vs off-peak)
  - Weekday vs weekend
  - "Night Shift" detection (A train on local tracks 23:00–06:00)

- [ ] **Data Quality**
  - Missing values
  - Duplicate arrivals
  - Gaps in service (overnight)

- [ ] **Sequence Length**
  - Validate 10-15 event lookback window
  - Session break detection (gaps > 120 min)

---

## Stage 3: ML Feature Engineering

After EDA confirms assumptions, apply the following transforms:

### Feature Vector (7 dimensions)

| Feature | Dim | Transformation |
|---------|-----|----------------|
| Log-Headway | 1 | `log(headway_minutes + 1)` |
| Time of Day | 2 | `sin(2π·S/86400)`, `cos(2π·S/86400)` where S = seconds since midnight |
| Train Identity | 3 | One-hot: A=[1,0,0], C=[0,1,0], E=[0,0,1] |
| Day of Week | 1 | Weekend flag (0/1) or cyclical encoding |

### Input Tensor Shape

```
(batch_size, lookback_window, num_features) = (32, 15, 7)
```

### Temporal Splits

| Split | Duration | Date Range | Purpose |
|-------|----------|------------|---------|
| Train | 1 month | Sep 13 – Oct 12, 2025 | Learn patterns |
| Validation | 1 month | Oct 13 – Nov 12, 2025 | Hyperparameter tuning |
| Test | 2 months | Nov 13, 2025 – Jan 13, 2026 | Final evaluation |

### Output Files

| File | Description |
|------|-------------|
| `data/a32_southbound.parquet` | A1 track features + split labels |
| `data/a32_northbound.parquet` | A2 track features + split labels |

---

## Stage 4: Model Architecture

### Multi-Head Stacked GRU

```
Input (15, 7)
  → GRU(128, return_sequences=True)
  → Dropout(0.2)
  → GRU(64)
  → Dropout(0.2)
  ├── Dense(1, linear)        → Time Output (Δt to next train)
  └── Dense(3, softmax)       → Type Output (A/C/E classification)
```

### Loss Functions

| Head | Output | Loss Function |
|------|--------|---------------|
| Time | Regression (1) | Huber Loss (robust to outliers) |
| Type | Classification (3) | Categorical Crossentropy |

### Rollout Inference

For line-specific predictions (e.g., "When is the next E train?"):
1. Model predicts next train: Δt=2min, Type=C
2. Simulate C arrival, update input window
3. Model predicts again: Δt=3min, Type=E
4. **Result:** "E train in 5 minutes"

---

## File Locations

| Asset | Path |
|-------|------|
| Raw data query script | `pipelines/sql/f11_eda_extract.sql` |
| EDA notebook | `notebooks/a32_eda.ipynb` |
| ML preprocessing script | `pipelines/preprocess_a32_ml.py` |
| Training script | `src/training/train_gru.py` |

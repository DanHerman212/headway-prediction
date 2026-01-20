# ML Pipeline Implementation Plan
## Headway Prediction - Production Training Pipeline

**Project:** Real-time Headway Prediction for NYC Subway  
**Target:** West 4th Street–Washington Square (A32) - A, C, E Lines  
**Platform:** Vertex AI with Experiments & TensorBoard  
**Date:** January 2026  

---

## Project Context

### Current State
- ✅ **Prediction Task Defined:** Multi-head Stacked GRU for train headway and type prediction
- ✅ **Historical Data:** Available in BigQuery `headway_prediction.clean` dataset
- ✅ **Real-time Pipeline:** GTFS ingestion delivering proper data representation
- ✅ **Production Reference:** Archive contains production-grade modules from previous project

### Objectives
1. Build production-ready ML training pipeline using Vertex AI
2. Implement comprehensive experiment tracking with TensorBoard
3. Create ML dataset with proper feature engineering (Stage 3)
4. Train and evaluate multi-head Stacked GRU model
5. Enable sharing of model performance insights with stakeholders

---

## Phase 1: ML Dataset Creation

### 1.1 SQL Transformation Script
**File:** `sql/02_create_ml_dataset.sql`

**Purpose:** Transform `headway_prediction.clean` → `headway_prediction.ml`

**Data Filters:**
- `stop_id IN ('A32N', 'A32S')`
- `route_id IN ('A', 'C', 'E')`
- `track IN ('A1', 'A2')` (local tracks only)
- Date range: Sep 13, 2025 → Jan 13, 2026 (4 months)

**Feature Engineering (Stage 3):**

| Feature | Dimensions | Transformation | Rationale |
|---------|------------|----------------|-----------|
| **Log-Headway** | 1 | `LOG(headway_minutes + 1)` | Normalize right-skewed distribution |
| **Time of Day** | 2 | `SIN(2π·S/86400)`, `COS(2π·S/86400)` | Cyclical time representation |
| **Train Identity** | 3 | One-hot: A=[1,0,0], C=[0,1,0], E=[0,0,1] | Categorical encoding |
| **Day of Week** | 1 | Weekend flag (0=weekday, 1=weekend) | Capture service pattern differences |

**Total Feature Dimensions:** 7

**Temporal Splits:**

| Split | Duration | Date Range | Proportion | Purpose |
|-------|----------|------------|------------|---------|
| Train | ~3 months | Sep 13 – Dec 12, 2025 | 60% | Learn patterns |
| Validation | ~1 month | Dec 13, 2025 – Jan 5, 2026 | 20% | Hyperparameter tuning |
| Test | ~2 weeks | Jan 6 – Jan 13, 2026 | 20% | Final evaluation |

**Output Schema:**
```sql
-- Base columns
route_id STRING
direction STRING  
stop_id STRING
track STRING
arrival_time_ts TIMESTAMP
day_type STRING

-- Sequence columns
prev_arrival_time TIMESTAMP
headway_minutes INT
headway_seconds_remainder INT

-- Engineered features
log_headway FLOAT64
time_sin FLOAT64
time_cos FLOAT64
train_is_a INT
train_is_c INT
train_is_e INT
is_weekend INT

-- Split label
data_split STRING  -- 'train', 'val', or 'test'
```

**Output Files:**
- `data/ml/a32_southbound.parquet` (Track A1)
- `data/ml/a32_northbound.parquet` (Track A2)

### 1.2 Execution Steps
1. Create SQL script with feature engineering logic
2. Execute against BigQuery `headway_prediction.clean` dataset
3. Verify row counts and feature distributions
4. Export results to GCS and/or local parquet files
5. Validate split proportions match expected ratios

---

## Phase 2: Exploratory Data Analysis

### 2.1 EDA Notebook
**File:** `notebooks/ml_dataset_eda.ipynb`

**Analysis Checklist:**

#### Feature Validation
- [ ] **Log-Headway Distribution**
  - Histogram should approximate normal distribution
  - Identify outliers (headways > 120 min)
  - Confirm mean/std in expected ranges
  
- [ ] **Time of Day Encoding**
  - Verify sine/cosine form proper circles (plot sin vs cos)
  - Check coverage across all 24 hours
  - Validate peak hour patterns visible

- [ ] **Train Type Balance**
  - Count by route_id (A vs C vs E)
  - Check for severe class imbalance
  - Analyze by time of day (night shift: A on local tracks)

- [ ] **Weekend vs Weekday**
  - Proportion of weekend observations
  - Headway distribution differences
  - Service frequency patterns

#### Temporal Analysis
- [ ] **Split Proportions**
  - Verify 60/20/20 train/val/test split
  - Check date boundaries align with plan
  - Confirm no data leakage across splits

- [ ] **Sequence Characteristics**
  - Distribution of sequence lengths
  - Identify service gaps (>120 min)
  - Determine optimal lookback window (target: 10-15 events)

#### Data Quality
- [ ] **Missing Values**
  - Check null counts per column
  - Validate completeness of feature engineering
  
- [ ] **Track Separation**
  - A1 (southbound) vs A2 (northbound) differences
  - Confirm rationale for separate models if needed

- [ ] **Composite Headway Logic**
  - Verify headway measured across all lines on same track
  - Check edge cases at service gaps

### 2.2 Deliverables
- EDA notebook with visualizations
- Summary of findings document
- Recommendations for model architecture adjustments
- Finalized sequence length and preprocessing parameters

---

## Phase 3: ML Pipeline Architecture

### 3.1 Module Structure

```
src/
├── config.py                    # Centralized configuration
├── data/
│   ├── __init__.py
│   └── dataset.py              # Data loading & sequence generation
├── models/
│   ├── __init__.py
│   └── model.py                # Multi-head Stacked GRU
├── training/
│   ├── __init__.py
│   └── trainer.py              # Training loop with Vertex AI
├── tracking/
│   ├── __init__.py
│   └── experiment.py           # Vertex AI Experiments wrapper
├── evaluator.py                # Model evaluation
└── metrics.py                  # Custom loss functions & metrics
```

### 3.2 Module Responsibilities

#### `config.py`
- Model hyperparameters (GRU units, dropout rates, lookback window)
- Training configuration (batch size, epochs, learning rate, optimizer)
- Feature dimensions and schema
- Data paths (GCS buckets, local cache directories)
- Vertex AI configuration (project ID, region, experiment name)
- TensorBoard logging settings

#### `data/dataset.py`
- Load parquet files from GCS or local storage
- Create sequences with sliding window
- Handle train/val/test split filtering
- Implement TensorFlow/PyTorch Dataset wrapper
- Batch generation with proper padding
- Data augmentation (if applicable)

#### `models/model.py`
- **Architecture:** Multi-head Stacked GRU
  ```
  Input: (batch_size, lookback_window, num_features) = (B, 15, 7)
  
  → GRU(128 units, return_sequences=True)
  → Dropout(0.2)
  → GRU(64 units, return_sequences=False)
  → Dropout(0.2)
  
  Head 1 (Time):  → Dense(1, activation='linear')    # Headway regression
  Head 2 (Type):  → Dense(3, activation='softmax')   # A/C/E classification
  ```
- Model compilation with custom loss weights
- Model summary and architecture visualization

#### `training/trainer.py`
- Initialize Vertex AI Experiments
- Setup TensorBoard callback
- Training loop with progress tracking
- Model checkpointing to GCS
- Early stopping based on validation loss
- Learning rate scheduling
- Metric logging (per epoch, per batch)
- Gradient monitoring

#### `tracking/experiment.py`
- Vertex AI Experiments SDK wrapper
- Parameter logging (hyperparameters, data config)
- Metric logging (loss, MAE, accuracy)
- Artifact upload (model weights, plots, reports)
- TensorBoard integration
- Experiment comparison utilities

#### `evaluator.py`
- Load trained model from checkpoint
- Evaluation on test set
- Per-line performance metrics (A vs C vs E)
- Rollout inference simulation (multi-step prediction)
- Visualization generation:
  - Actual vs predicted plots
  - Error distribution histograms
  - Confusion matrix for train type
  - Per-hour performance breakdown
- Generate evaluation report

#### `metrics.py`
- **Time Head Metrics:**
  - Huber Loss (robust to outliers)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  
- **Type Head Metrics:**
  - Categorical Crossentropy
  - Accuracy
  - Per-class F1 scores
  - Confusion matrix
  
- **Combined Multi-task Loss:**
  - Weighted sum: `λ_time * L_time + λ_type * L_type`

---

## Phase 4: Implementation Details

### 4.1 Model Architecture Specifications

**Input Tensor:**
- Shape: `(batch_size, lookback_window, num_features)`
- Example: `(32, 15, 7)`

**Feature Order:**
1. `log_headway` (1D)
2. `time_sin` (1D)
3. `time_cos` (1D)
4. `train_is_a` (1D)
5. `train_is_c` (1D)
6. `train_is_e` (1D)
7. `is_weekend` (1D)

**Target Outputs:**
- **Time Head:** Next train headway (continuous, minutes)
- **Type Head:** Next train type (categorical, A/C/E)

**Loss Functions:**
- Time: Huber Loss (δ=1.0) - robust to outliers
- Type: Categorical Crossentropy
- Combined: `0.7 * L_time + 0.3 * L_type` (weights tunable)

### 4.2 Training Configuration

**Hyperparameters (Initial):**
```python
LOOKBACK_WINDOW = 15        # Number of previous events
BATCH_SIZE = 32             # Training batch size
EPOCHS = 100                # Maximum epochs
LEARNING_RATE = 1e-3        # Adam optimizer
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.2

GRU_UNITS_1 = 128           # First GRU layer
GRU_UNITS_2 = 64            # Second GRU layer

LOSS_WEIGHT_TIME = 0.7
LOSS_WEIGHT_TYPE = 0.3
```

**Optimizer:** Adam with learning rate decay
**Validation Frequency:** Every epoch
**Checkpoint Strategy:** Save best model based on validation loss

### 4.3 Vertex AI Integration

**Experiment Setup:**
```python
from google.cloud import aiplatform

aiplatform.init(
    project="YOUR_PROJECT_ID",
    location="us-central1",
    experiment="headway-prediction-gru"
)

aiplatform.start_run(run="run-{timestamp}")
```

**TensorBoard Configuration:**
```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f"gs://{bucket}/tensorboard/{run_id}",
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)
```

**Metrics to Log:**
- Training loss (time, type, combined)
- Validation loss (time, type, combined)
- MAE, RMSE for time predictions
- Accuracy, F1 for type predictions
- Learning rate
- Gradient norms

### 4.4 Data Pipeline

**Sequence Generation Logic:**
```python
def create_sequences(df, lookback=15, min_gap_minutes=120):
    """
    Create sequences with sliding window.
    Break sequences when gap > min_gap_minutes (service interruption).
    """
    sequences = []
    targets_time = []
    targets_type = []
    
    for gap in detect_service_gaps(df, min_gap_minutes):
        segment = df[gap.start:gap.end]
        for i in range(lookback, len(segment)):
            X = segment.iloc[i-lookback:i][FEATURE_COLS].values
            y_time = segment.iloc[i]['headway_minutes']
            y_type = segment.iloc[i]['route_id']  # A/C/E
            
            sequences.append(X)
            targets_time.append(y_time)
            targets_type.append(y_type)
    
    return sequences, targets_time, targets_type
```

### 4.5 Evaluation Strategy

**Metrics by Category:**

1. **Overall Performance:**
   - Combined loss on test set
   - Time MAE (minutes)
   - Type accuracy (%)

2. **Per-Line Performance:**
   - MAE for A-train predictions
   - MAE for C-train predictions  
   - MAE for E-train predictions

3. **Temporal Performance:**
   - Peak hours (7-9 AM, 5-7 PM)
   - Off-peak hours
   - Overnight (night shift mode)

4. **Rollout Inference:**
   - Multi-step prediction accuracy
   - "Next E train in X minutes" scenarios
   - Cumulative error over prediction horizon

**Visualization Requirements:**
- Actual vs Predicted scatter plots
- Residual distribution histograms
- Per-hour error box plots
- Train type confusion matrix
- TensorBoard scalar plots

---

## Phase 5: Testing & Deployment

### 5.1 Local Testing
1. **Unit Tests:** Test each module independently
2. **Integration Test:** Run end-to-end with small data subset
3. **TensorBoard Validation:** Verify logs generated correctly
4. **Checkpoint Validation:** Verify model saving/loading

### 5.2 Vertex AI Training Job

**Container Setup:**
```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
WORKDIR /app
```

**Submit Training:**
```python
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name="headway-gru-training",
    container_uri=f"gcr.io/{project}/headway-trainer:latest",
    requirements=["tensorflow==2.13", "google-cloud-aiplatform"],
)

job.run(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=["--epochs=100", "--batch-size=32"],
    tensorboard=tensorboard_resource_name,
)
```

### 5.3 Model Registry
1. Register best model to Vertex AI Model Registry
2. Tag with version and metadata
3. Link to experiment run for reproducibility

### 5.4 Deployment (Future Phase)
- Deploy to Vertex AI Prediction endpoint
- Configure auto-scaling
- Monitor prediction latency
- Integrate with real-time GTFS pipeline

---

## Phase 6: Deliverables & Success Criteria

### Deliverables
1. **SQL Script:** Feature engineering transformation
2. **ML Dataset:** Parquet files with engineered features
3. **EDA Notebook:** Comprehensive data analysis with visualizations
4. **Python Modules:** Production-ready codebase
5. **Trained Model:** Checkpointed model in GCS
6. **Evaluation Report:** Model performance metrics and insights
7. **TensorBoard Logs:** Interactive dashboards for stakeholders
8. **Documentation:** Implementation guide and usage instructions

### Success Criteria

**Data Quality:**
- ✅ Features correctly engineered per specification
- ✅ No data leakage between train/val/test splits
- ✅ Temporal splits align with date boundaries

**Model Performance:**
- 🎯 Time MAE < 2.0 minutes (stretch: < 1.5 min)
- 🎯 Type Accuracy > 85% (stretch: > 90%)
- 🎯 Rollout inference maintains accuracy over 3-step horizon

**Production Readiness:**
- ✅ All modules tested and documented
- ✅ Vertex AI integration functional
- ✅ TensorBoard dashboards accessible to stakeholders
- ✅ Model artifacts versioned and reproducible
- ✅ Evaluation reports generated automatically

**Stakeholder Enablement:**
- ✅ TensorBoard shared with ability to compare experiments
- ✅ Clear visualizations of model performance
- ✅ Per-line metrics available (A vs C vs E)
- ✅ Documentation enables handoff and iteration

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. ML Dataset Creation | 1-2 days | Access to BigQuery |
| 2. EDA | 1 day | Phase 1 complete |
| 3. Architecture Design | 0.5 days | Phase 2 insights |
| 4. Module Implementation | 3-4 days | Phase 3 design |
| 5. Testing & Training | 2-3 days | Phase 4 complete |
| 6. Evaluation & Documentation | 1-2 days | Phase 5 complete |
| **Total** | **8-12 days** | |

---

## Risk Mitigation

**Risk:** Data quality issues in `clean` dataset
- **Mitigation:** Thorough EDA before proceeding to model training

**Risk:** Class imbalance in train types
- **Mitigation:** Use weighted loss functions or sampling strategies

**Risk:** Sequence gaps disrupt model training
- **Mitigation:** Explicit gap detection logic with 120-min threshold

**Risk:** Model overfits to training period
- **Mitigation:** Strong validation strategy, early stopping, dropout regularization

**Risk:** Vertex AI integration complexity
- **Mitigation:** Local testing first, incremental integration

---

## Next Steps

1. ✅ Document implementation plan (this file)
2. ⏭️ **Create SQL transformation script** (`sql/02_create_ml_dataset.sql`)
3. ⏭️ Execute SQL and generate ML dataset
4. ⏭️ Build EDA notebook
5. ⏭️ Implement Python modules
6. ⏭️ Train and evaluate model

---

## References

- **Data Representation Guide:** `archive/docs/data_representation.md`
- **Archive Modules:** `archive/backup/src/`
- **Previous Project Config:** `archive/backup/src/config.py`
- **Previous Trainer:** `archive/backup/src/training/trainer.py`

---

**Document Version:** 1.0  
**Last Updated:** January 20, 2026  
**Author:** ML Engineering Team

# NYC Subway Headway Prediction - Project Plan
**Date:** January 23, 2026  
**Prediction Task:** Real-time headway prediction at W4 Washington Square station  
**Architecture:** Multi-head Stacked GRU (Regression + Classification)

---

## 🎯 Project Understanding

### Universal ML Framework Goal
Build a reusable framework with comprehensive Vertex AI Experiments and TensorBoard integration that can handle:
- **Temporal data** (current project)
- **Images** (future)
- **Natural Language** (future)

### Current Prediction Task: NYC Subway Headways

**Station:** W4 Washington Square  
**Track:** Southbound local track  
**Routes:** A, C, E trains

#### Service Patterns (Complex Scheduling)
1. **Peak Hours (Weekdays)**:
   - A train: Express service
   - C & E trains: Local service

2. **Late Night (10pm - 5am)**:
   - C train: Terminates at 10pm
   - A train: Switches from express to local
   - Continues until ~5am

3. **Early Morning (5am+)**:
   - Normal service resumes

#### Why Two Models?
- **Model 1**: Local service patterns (C, E, late-night A)
- **Model 2**: Express service patterns (peak A)

---

## 🏗️ Model Architecture

### Multi-Head Output Architecture

```
Input Sequence (20 timesteps)
    ↓
[Composite headways: E→C→C→E or A→C→A→C]
    ↓
GRU Layer 1 (128 units)
    ↓
GRU Layer 2 (64 units)
    ↓
    ├─→ [Regression Head]  → Headway prediction (seconds)
    │   Loss: Huber
    │   Metric: MAE seconds
    │
    └─→ [Classification Head] → Route ID probability (A/C/E)
        Loss: Sparse Categorical Cross-Entropy
        Metric: Accuracy
```

### Model Specifications
- **Lookback Window (L):** 20 timesteps
- **Forecast Horizon (H):** Next event (1 timestep)
- **Regression Output:** Next headway in seconds
- **Classification Output:** Probability distribution over route_ids [A, C, E]

---

## 📊 Data Pipeline

### Data Source
- **Platform:** BigQuery
- **Dataset:** ML dataset (existing)
- **Filter Requirements:**
  - Station: W4 Washington Square
  - Direction: Southbound
  - Track: Local
  - Routes: A, C, E

### Preprocessing Steps

#### Phase 1: Core Preprocessing (Start Here)
1. **Log-scale headways**
   ```python
   headway_log = np.log1p(headway_seconds)
   ```

2. **One-hot encode route_ids**
   ```python
   route_id → [A, C, E] → one-hot vectors
   ```

3. **Temporal features (daily/weekly periodicity)**
   ```python
   # Daily cycle (24 hours)
   hour_sin = sin(2π * hour / 24)
   hour_cos = cos(2π * hour / 24)
   
   # Weekly cycle (7 days)
   day_sin = sin(2π * day_of_week / 7)
   day_cos = cos(2π * day_of_week / 7)
   ```

#### Phase 2: Advanced Features (Later)
4. **Rolling regime statistics**
   - std_10, std_50: Standard deviation over 10 and 50 events
   - mean_10, mean_50: Rolling mean over 10 and 50 events
   - max_50: Maximum over 50 events
   - Purpose: Detect irregular shocks/disruptions

5. **Weather data integration** (Future enhancement)

### Data Representation
- **Input shape:** `(batch_size, 20, num_features)`
  - 20 timesteps (lookback window)
  - Features: [log_headway, route_one_hot (3), hour_sin, hour_cos, day_sin, day_cos]
  
- **Output shape:**
  - Regression: `(batch_size, 1)` - next headway
  - Classification: `(batch_size, 3)` - route_id probabilities

---

## 🎓 Training Strategy

### Run 1: Overfit to Beat Baseline
**Goal:** Defeat common-sense baseline, prove model can learn

**Configuration:**
- No regularization
- Higher learning rate
- Goal: Overfit intentionally
- Metrics to beat:
  - MAE seconds: < baseline (mean predictor)
  - Accuracy: > 1/3 (random guess for 3 classes)

**Success Criteria:**
- Training MAE significantly better than validation
- Training accuracy → 100%
- Confirms model has sufficient capacity

### Run 2: Regularization & Generalization
**Goal:** Reduce overfitting, improve generalization

**Techniques:**
- Dropout layers
- L2 regularization
- Learning rate decay
- Early stopping

**Success Criteria:**
- Validation metrics improve
- Training/validation gap reduces
- Better real-world performance

### Run 3+: Hyperparameter Optimization (Future)
- Experiment with GRU units
- Adjust lookback window
- Try different loss weightings

---

## 🚀 Deployment Pipeline

### Stage 1: Model Development (Local)
1. Test data pipeline locally
2. Train and validate model
3. Full experiment tracking with TensorBoard

### Stage 2: Cloud Training (Vertex AI)
1. Move to Vertex AI Training
2. Experiment tracking continues seamlessly
3. Compare local vs cloud runs

### Stage 3: Model Deployment
1. **Deploy to Vertex AI Prediction Endpoint**
   - Best model from experiments
   - Low-latency inference

2. **Integration with Live Event Stream**
   - Real-time predictions on incoming events
   - Store predictions in NoSQL database

3. **Mobile App Integration**
   - Serve predictions to commuters
   - Reduce commute uncertainty
   - User-facing feature

---

## 📝 Execution Plan - Step by Step

### Phase 1: Data Foundation (Days 1-2)
**Steps:**
- [ ] 1.1: Create BigQuery query to extract W4 data
- [ ] 1.2: Explore data, understand patterns
- [ ] 1.3: Implement log-scaling for headways
- [ ] 1.4: Implement one-hot encoding for route_ids
- [ ] 1.5: Generate temporal features (sin/cos)
- [ ] 1.6: Validate preprocessing locally
- [ ] 1.7: Create train/val/test splits (60/20/20)

**Deliverable:** Clean, preprocessed dataset ready for model

---

### Phase 2: Data Loading (Day 3)
**Steps:**
- [ ] 2.1: Create windowing function (L=20, H=1)
- [ ] 2.2: Build tf.data.Dataset pipeline
- [ ] 2.3: Implement batching and shuffling
- [ ] 2.4: Test data shapes and formats
- [ ] 2.5: Optimize pipeline with prefetching

**Deliverable:** Efficient tf.data.Dataset for training

---

### Phase 3: Model Architecture (Day 4)
**Steps:**
- [ ] 3.1: Create multi-head GRU model class
- [ ] 3.2: Implement regression head (Huber loss)
- [ ] 3.3: Implement classification head (Sparse CE)
- [ ] 3.4: Define custom metrics (MAE seconds, Accuracy)
- [ ] 3.5: Test model with dummy data
- [ ] 3.6: Verify output shapes

**Deliverable:** Working multi-head GRU architecture

---

### Phase 4: Training Infrastructure (Day 5)
**Steps:**
- [ ] 4.1: Configure ModelConfig for this task
- [ ] 4.2: Configure TrackingConfig for experiments
- [ ] 4.3: Initialize ExperimentTracker
- [ ] 4.4: Set up Trainer with multi-loss support
- [ ] 4.5: Test end-to-end training loop locally

**Deliverable:** Complete training pipeline with tracking

---

### Phase 5: Baseline & Run 1 (Days 6-7)
**Steps:**
- [ ] 5.1: Calculate baseline metrics (mean predictor)
- [ ] 5.2: Run 1 - Overfit configuration
- [ ] 5.3: Monitor TensorBoard (scalars, histograms, graphs)
- [ ] 5.4: Analyze overfitting behavior
- [ ] 5.5: Confirm model can learn patterns

**Deliverable:** Overfitted model beating baseline

---

### Phase 6: Run 2 - Regularization (Days 8-9)
**Steps:**
- [ ] 6.1: Add dropout layers
- [ ] 6.2: Add L2 regularization
- [ ] 6.3: Implement learning rate decay
- [ ] 6.4: Train with early stopping
- [ ] 6.5: Compare runs in TensorBoard
- [ ] 6.6: Evaluate on test set

**Deliverable:** Generalized model ready for deployment

---

### Phase 7: Vertex AI Migration (Day 10)
**Steps:**
- [ ] 7.1: Create Dockerfile for training
- [ ] 7.2: Build and push to Artifact Registry
- [ ] 7.3: Configure Vertex AI Training job
- [ ] 7.4: Submit training job
- [ ] 7.5: Monitor Vertex AI Experiments
- [ ] 7.6: Verify TensorBoard integration

**Deliverable:** Cloud-based training pipeline

---

### Phase 8: Model Deployment (Days 11-12)
**Steps:**
- [ ] 8.1: Export best model for serving
- [ ] 8.2: Create Vertex AI Endpoint
- [ ] 8.3: Deploy model to endpoint
- [ ] 8.4: Test prediction latency
- [ ] 8.5: Integrate with live event stream
- [ ] 8.6: Set up NoSQL storage for predictions

**Deliverable:** Production prediction endpoint

---

### Phase 9: Monitoring & Iteration (Ongoing)
**Steps:**
- [ ] 9.1: Monitor prediction quality
- [ ] 9.2: Track model drift
- [ ] 9.3: Collect user feedback
- [ ] 9.4: Plan Phase 2 features (rolling stats, weather)

**Deliverable:** Production-ready system with monitoring

---

## 🔧 Technical Stack

### Framework Components
```
ml_pipelines/
├── config/                     # Single source of truth
│   ├── model_config.py         # Model hyperparameters
│   └── tracking_config.py      # Experiment tracking config
│
├── data/
│   ├── bigquery_etl.py         # BQ extraction
│   ├── preprocessing.py        # NEW: Headway preprocessing
│   └── dataset_generator.py   # tf.data.Dataset creation
│
├── models/
│   ├── multi_head_gru.py       # NEW: Multi-head GRU architecture
│   └── base_model.py           # Base class (existing)
│
├── training/
│   └── trainer.py              # Multi-loss training support
│
├── evaluation/
│   ├── metrics.py              # Custom metrics
│   └── evaluator.py            # Multi-head evaluation
│
└── tracking/
    ├── tracker.py              # ⭐ Vertex AI + TensorBoard
    └── callbacks.py            # Auto-logging
```

### Key Technologies
- **Data:** BigQuery, Pandas, NumPy
- **ML Framework:** TensorFlow/Keras
- **Architecture:** Stacked GRU (RNN)
- **Tracking:** Vertex AI Experiments + TensorBoard
- **Deployment:** Vertex AI Prediction Endpoints
- **Storage:** NoSQL database (predictions)

---

## 📊 Experiment Tracking Details

### What Gets Logged (Automatically)
1. **Scalars** (every epoch):
   - Total loss, regression loss, classification loss
   - MAE seconds (regression)
   - Accuracy (classification)
   - Learning rate

2. **Histograms** (every epoch):
   - GRU weight distributions
   - Gradient flows
   - Layer activations

3. **Graphs** (once):
   - Multi-head model architecture
   - Data flow visualization

4. **HParams** (once):
   - GRU units [128, 64]
   - Learning rate
   - Dropout rate
   - Batch size
   - Loss weights

5. **Custom Visualizations**:
   - Prediction vs actual headways
   - Route ID confusion matrix
   - Temporal pattern analysis

### Comparing Runs
All runs logged to Vertex AI Experiments enable:
- Side-by-side comparison of Run 1 vs Run 2
- Hyperparameter impact analysis
- Model selection based on metrics

---

## ✅ Success Criteria

### Phase 1 Success (Data Pipeline)
- ✅ Clean extraction from BigQuery
- ✅ Correct preprocessing (log-scale, one-hot, temporal)
- ✅ Valid train/val/test splits
- ✅ tf.data.Dataset with correct shapes

### Phase 2 Success (Model Training)
- ✅ Run 1: Overfit beats baseline
- ✅ Run 2: Generalization improves
- ✅ Full TensorBoard logging working
- ✅ Vertex AI Experiments tracking active

### Phase 3 Success (Deployment)
- ✅ Model deployed to endpoint
- ✅ Latency < 100ms for predictions
- ✅ Integration with event stream working
- ✅ Predictions served to mobile app

---

## 🚨 Risk Mitigation

### Risk 1: Data Quality Issues
**Mitigation:**
- Extensive EDA before modeling
- Data validation at each preprocessing step
- Outlier detection and handling

### Risk 2: Model Doesn't Learn
**Mitigation:**
- Start with overfit attempt (Run 1)
- Simplify architecture if needed
- Debug with small dataset first

### Risk 3: Local vs Cloud Differences
**Mitigation:**
- Test locally first (containerized)
- Use same Docker image for local and cloud
- Verify environment parity

### Risk 4: Deployment Latency
**Mitigation:**
- Profile model inference time
- Optimize if needed (model quantization, pruning)
- Test under load before production

---

## 📚 Next Immediate Action

**Start with Phase 1, Step 1.1:**
```python
# Create BigQuery extraction query for W4 station
# Extract: timestamp, route_id, headway, direction, stop_id
# Filter: W4 Washington Square, Southbound, Local track
```

**Command:**
```bash
# Navigate to ml_pipelines
cd ml_pipelines

# Create new module for this project
mkdir -p data/subway_headway
touch data/subway_headway/__init__.py
touch data/subway_headway/bigquery_extractor.py
touch data/subway_headway/preprocessor.py
```

---

## 💡 Key Principles for Success

1. **Go Slow:** Test each component independently
2. **Validate Often:** Check data shapes at every step
3. **Track Everything:** Full experiment logging from day 1
4. **Test Locally First:** Prove it works before Vertex AI
5. **Single Source of Truth:** All configs in ModelConfig/TrackingConfig
6. **Iterate:** Run 1 → Run 2 → Run 3 with learnings

---

## 📞 Questions Before We Start

1. **BigQuery Access:** Do you have the table name and schema?
2. **Date Range:** How much historical data should we extract?
3. **Vertex AI Setup:** Project ID and region configured?
4. **TensorBoard Instance:** Do you have a Vertex AI TensorBoard instance, or shall we create one?
5. **Baseline Metric:** What's the acceptable MAE threshold for production?

---

**Status:** Ready to begin Phase 1, Step 1.1  
**Next Step:** Create BigQuery extraction query  
**Tracking:** Full Vertex AI + TensorBoard from first run ✅

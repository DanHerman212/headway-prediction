# A1 Track Headway Prediction Model

Modular pipeline for training a stacked GRU model to predict train headways on the A1 (local) track at West 4th St station.

## Project Structure

```
A1/
├── src/                # Python modules
│   ├── config.py       # Configuration and hyperparameters
│   ├── extract_data.py # BigQuery data extraction → CSV
│   ├── preprocess.py   # Feature engineering and encoding
│   ├── model.py        # Stacked GRU with multi-output architecture
│   ├── train.py        # Training pipeline with TensorBoard/Vertex AI
│   └── evaluate.py     # Evaluation metrics and visualizations
├── pipeline.py         # Vertex AI Pipelines (KFP) definition
├── Dockerfile          # Container image for Vertex AI
├── cloudbuild.yaml     # Cloud Build configuration
├── deploy.sh           # Automated deployment script
├── test_modules.py     # Module validation tests
├── requirements.txt    # Python dependencies
└── README.md
```

## Pipeline Overview

### 1. Data Extraction
```bash
python extract_data.py --output data/A1/raw_data.csv
```
- Queries BigQuery for A1 track data
- Saves as CSV artifact (52k+ records)
- Columns: `arrival_time, headway, route_id, track, time_of_day, day_of_week`

### 2. Preprocessing
```bash
python preprocess.py --input data/A1/raw_data.csv --output data/A1/preprocessed_data.npy
```
**Transformations:**
- **Headway:** Log transformation `log(headway + 1)` for outlier handling
- **Route ID:** One-hot encode (A, C, E) → 3 binary features
- **Temporal:** Cyclical encoding
  - Daily: `hour_sin = sin(2π·hour/24)`, `hour_cos`
  - Weekly: `dow_sin = sin(2π·dow/7)`, `dow_cos`

**Output:** Numpy array `(n_samples, 8)` with features:
```
[log_headway, route_A, route_C, route_E, hour_sin, hour_cos, dow_sin, dow_cos]
```

### 3. Training
```bash
python train.py --run_name exp01-baseline
```

**Model Architecture:** Stacked GRU with Multi-Output
```
Input (batch, 20, 8)  # 20 timesteps, 8 features
  ↓
GRU Layer 1 (128 units, return_sequences=True)
  ↓
Dropout (0.2)
  ↓
GRU Layer 2 (64 units, return_sequences=False)
  ↓
Dropout (0.2)
  ↓
Dense (32 units, relu)
  ↓
       ┌─────────────┴─────────────┐
       ↓                           ↓
Dense (3, softmax)          Dense (1, linear)
[Route Classification]      [Headway Regression]
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Losses: Categorical crossentropy (route), Huber (headway)
- Metrics: Accuracy (route), MAE seconds (headway)
- Callbacks: TensorBoard (scalars, histograms, graph, profiling), ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

**Data Splits:** Chronological (60% train, 20% val, 20% test)

**TensorBoard Tracking:**
- Scalars: loss, metrics, learning rate
- Histograms: weight distributions, gradients
- Graph: model architecture
- HParams: hyperparameter comparison
- Profiling: GPU/CPU performance (batches 10-20)

**Vertex AI Experiments:**
- Automatic experiment creation
- Run tracking with hyperparameters
- Metric logging
- TensorBoard integration

### 4. Evaluation
```bash
python evaluate.py --run_name exp01-baseline
```

**Metrics:**
- **Classification:** F1 score (macro/weighted) for route prediction
- **Regression:** MAE/RMSE in seconds for headway prediction
- **Baseline:** Persistence model comparison

**Visualizations:**
- Confusion matrix (route classification)
- Predicted vs actual (scatter)
- Residual plot
- Error distribution
- Time series overlay

## Key Design Decisions

### Multi-Output Architecture
- **Why:** Station receives trains from routes A, C, E (composite headways)
- **Classification head:** Learns which route arrives next
- **Regression head:** Learns headway timing
- **Benefit:** Joint optimization captures route-specific headway patterns

### Log Transformation
- **From EDA:** 170x skewness reduction with `log(x + 1)`
- **Handles outliers** naturally (max headway 2655 min → ~7.88 in log space)
- **Inverse transform** for predictions: `exp(pred) - 1`

### Cyclical Temporal Encoding
- **Daily/weekly periodicity** captured via sin/cos
- **Avoids discontinuity** at boundaries (hour 23 → 0, Sunday → Monday)

### Chronological Splits
- **Prevents data leakage** in time series
- **Realistic evaluation** on future unseen data

## Model Rationale (A1 Track)

From EDA autocorrelation analysis:
- **Weak autocorrelation** (5.37% max ACF)
- **Schedule-driven behavior** (0.3% variance explained by lag-1)
- **Lookback: 20 events** captures sufficient context
- **Temporal features dominate** → GRU processes sequential patterns

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Extract data from BigQuery
python extract_data.py

# 2. Preprocess features
python preprocess.py

# 3. Train model
python train.py --run_name my_first_run

# 4. Evaluate on test set
python evaluate.py --run_name my_first_run

# 5. View TensorBoard
tensorboard --logdir gs://st-convnet-training-configuration/tensorboard/A1
```

## Configuration

Edit `config.py` to adjust hyperparameters:
- Lookback window, forecast horizon
- Batch size, learning rate, epochs
- GRU units, dropout rate
- Data paths, BigQuery settings
- TensorBoard/Vertex AI settings

## Output Artifacts

**Training:**
- `models/A1/checkpoints/best_model.keras` - Best model checkpoint
- `models/A1/checkpoints/{run_name}_history.json` - Training history
- `models/A1/checkpoints/{run_name}_config.json` - Config snapshot
- `models/A1/checkpoints/{run_name}_training_log.csv` - Epoch-by-epoch logs
- `gs://.../tensorboard/A1/{run_name}/` - TensorBoard logs

**Evaluation:**
- `models/A1/checkpoints/{run_name}_evaluation/evaluation_results.json`
- `confusion_matrix.png`
- `predicted_vs_actual.png`
- `residuals.png`
- `error_distribution.png`
- `time_series_overlay.png`

## Next Steps

1. **Test architecture** with `python model.py` (validates model builds correctly)
2. **Run pipeline** end-to-end on sample data
3. **Compare runs** in TensorBoard (hyperparameter sweep)
4. **Deploy to Vertex AI Pipelines** for production training

# NYC Subway Headway Prediction

Real-time headway prediction for A, C, E subway lines at West 4th Street-Washington Square station.

---

## Table of Contents

1. [Define the Prediction Task](#1-define-the-prediction-task)
2. [Build a Data Representation](#2-build-a-data-representation)
3. [Build an ML System](#3-build-an-ml-system)
4. [Deploy the Model to Production](#4-deploy-the-model-to-production)

---

## 1. Define the Prediction Task

**Objective:** Predict headways (time between consecutive train arrivals) for A, C, E lines at West 4th Street station.

**Purpose:** This is a proof-of-concept to understand the spatiotemporal dynamics of a high-capacity train station that processes 150,000+ trips per year. The goal is to:
- **Reduce uncertainty** for passengers planning their commute
- **Improve operational performance** by helping dispatchers optimize train spacing
- **Validate ML approaches** for real-time transit prediction at complex urban hubs

**Station Context:** West 4th Street (A32S - Southbound) serves both local (A1 track) and express (A3 track) trains, making it an ideal test case for multi-track prediction models.

---

## 2. Build a Data Representation

**Dataset:** 75,390 train arrival events collected over 6 months (July 2025 - January 2026)

**Data Source:** NYC MTA real-time GTFS feeds and historical subway data archive

**Features:**
- `headway` - Time between consecutive arrivals (target variable)
- `route_id` - Train route (A, C, or E)
- `track` - Platform track (A1 local, A3 express)
- `hour_of_day`, `day_of_week` - Temporal features
- `time_of_day_seconds` - Continuous time representation

**Data Pipeline:**
1. Download historical GTFS data from MTA archive
2. Clean and transform raw arrivals in BigQuery
3. Engineer temporal features and calculate headways
4. Split into track-specific datasets (A1: 52k events, A3: 23k events)

See [sql/02_feature_engineering.sql](sql/02_feature_engineering.sql) for full transformation logic.

---

## 3. Build an ML System

**Modeling Strategy:**

1. **Establish Common Sense Baseline**
   - Simple heuristics: median headway by hour, day-of-week averages
   - Naive persistence: use previous headway as prediction

2. **Choose Model Architecture**
   - Stacked GRU (Gated Recurrent Units) for sequence modeling
   - Separate models for A1 (local) and A3 (express) tracks due to distinct patterns
   - Input: 15-event lookback window with 7 features per event

3. **Scale Up to Overfit**
   - Start with large capacity (128→64 hidden units)
   - Train on full dataset without regularization
   - Verify model can learn complex temporal patterns

4. **Regularize for Robust Fit**
   - Add dropout layers (0.2) between GRU stacks
   - Early stopping on validation set
   - Use Huber loss for robustness to outliers

**Training Infrastructure:**
- Google Cloud Vertex AI for experiment tracking
- TensorBoard for performance visualization
- Temporal train/validation/test split (60/20/20)

See [notebooks/ml_eda.ipynb](notebooks/ml_eda.ipynb) for exploratory data analysis.

---

## 4. Deploy the Model to Production

**Deployment Pipeline:**
1. Register trained models to Vertex AI Model Registry
2. Deploy prediction endpoints with auto-scaling
3. Integrate with real-time GTFS ingestion pipeline

**Evaluation Metrics:**
- **Prediction Accuracy:** MAE, RMSE for headway estimates
- **Uncertainty Reduction:** Compare predicted vs observed variance
- **Operational Impact:** Measure how predictions improve passenger wait time estimates

**Monitoring:**
- Real-time prediction latency tracking
- Model drift detection on live GTFS feeds
- A/B testing framework for model iterations

---

## Project Structure

```
headway-prediction/
├── sql/              # BigQuery feature engineering
├── notebooks/        # EDA and analysis (ml_eda.ipynb)
├── python/           # Data ingestion scripts
├── pipelines/        # Airflow/Cloud Run pipeline configs
├── bash/             # Deployment utilities
└── archive/          # Reference implementations
```

---

## Data Sources

| Source | Volume | Location |
|--------|--------|----------|
| Historic Arrivals | ~10MB/day | MTA GTFS Archive |
| Train Schedules | ~9GB | `gs://bucket/raw/schedules/` |
| Station Metadata | ~40MB | Static GTFS feeds |

---

## Contributing

This is a research project. For questions or collaboration, please open an issue.

---

## License

MIT License - See LICENSE file for details

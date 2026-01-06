# Next Steps: Model Optimization & Experiment Framework

**Date:** January 6, 2026  
**Status:** Ready for tomorrow's session

---

## Overview

This plan covers three areas:
1. **Model Optimization** - Strategies to improve accuracy if below production thresholds
2. **Experiment Framework** - Horizon-specific metrics matching the paper's analysis
3. **Narrative Visualizations** - Meaningful spatiotemporal predictions for specific scenarios

---

## 1. Model Optimization Strategies

### Current Baseline
- **Architecture:** V1 ConvLSTM (371K params)
- **Target:** RMSE ≤ 90 seconds, R² ≥ 0.90

### If Accuracy is Below Threshold

#### A. Hyperparameter Tuning
| Parameter | Current | Try |
|-----------|---------|-----|
| Learning Rate | 0.001 | 0.0005, 0.0001 (slower convergence, finer tuning) |
| Batch Size | 128 | 64 (more gradient updates per epoch) |
| Filters | 32/64/32 | 64/128/64 (more capacity) |
| Kernel Size | (3,3) | (5,5) (larger receptive field) |

#### B. Architecture Modifications
1. **Deeper encoder:** Add 3rd ConvLSTM layer before bottleneck
2. **Attention mechanism:** Add spatial attention after bottleneck
3. **Residual connections:** Skip connection from encoder to decoder
4. **Dropout:** Add `SpatialDropout3D(0.2)` for regularization

#### C. Data Augmentation
- **Temporal jittering:** Shift windows by ±1-2 minutes
- **Noise injection:** Add Gaussian noise (σ=0.01) to inputs
- **Mixup:** Blend nearby time windows

#### D. Loss Function Experiments
```python
# Current: MSE
# Try: Huber loss (robust to outliers)
loss = tf.keras.losses.Huber(delta=1.0)

# Try: Weighted MSE (penalize peak hours more)
def weighted_mse(y_true, y_pred, weights):
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))
```

#### E. Ensemble Methods
- Train 3-5 models with different random seeds
- Average predictions for more stable results

---

## 2. Experiment Framework: Horizon-Specific Metrics

### Paper Reference (Usama & Koutsopoulos 2025)
The paper reports metrics broken down by forecast horizon (Table 2):
- t+1, t+5, t+10, t+15 minute predictions
- Shows how error grows with prediction horizon

### Implementation Plan

#### A. Create `src/experiments/horizon_analysis.py`

```python
class HorizonAnalyzer:
    """Analyze model performance across different forecast horizons."""
    
    def __init__(self, model, scaler, config):
        self.model = model
        self.scaler = scaler
        self.config = config
    
    def compute_horizon_metrics(self, test_dataset):
        """
        Compute RMSE and R² for each forecast timestep (t+1 to t+15).
        
        Returns:
            DataFrame with columns: horizon, rmse_seconds, r_squared
        """
        # Collect all predictions and targets
        # Compute metrics per timestep
        pass
    
    def plot_horizon_degradation(self, metrics_df, save_path=None):
        """
        Bar chart showing RMSE vs forecast horizon.
        Answers: "How much does accuracy degrade for longer predictions?"
        """
        pass
    
    def compare_models(self, models_dict, test_dataset):
        """
        Compare multiple models across horizons.
        Useful for ablation studies.
        """
        pass
```

#### B. Metrics to Report (matching paper Table 2)

| Horizon | RMSE (seconds) | R² | Notes |
|---------|---------------|-----|-------|
| t+1 | ? | ? | Immediate next minute |
| t+5 | ? | ? | Short-term |
| t+10 | ? | ? | Medium-term |
| t+15 | ? | ? | Full forecast horizon |
| **Average** | ? | ? | Overall performance |

#### C. Additional Analyses

1. **Peak vs Off-Peak Performance**
   - Morning rush (7-9 AM)
   - Evening rush (5-7 PM)
   - Off-peak hours
   
2. **Weekday vs Weekend**
   - Service patterns differ significantly

3. **Station-Specific Performance**
   - Terminal stations vs mid-line stations
   - High-traffic vs low-traffic stations

---

## 3. Narrative Visualizations

### Goal
Create compelling visualizations that tell a story, not just random samples.

### Scenario-Based Visualizations

#### A. Rush Hour Prediction
```python
# Find a morning rush hour sample (8:00 AM on a weekday)
scenario = {
    'name': 'Morning Rush - Penn Station',
    'time': '2025-03-15 08:00:00',
    'station_focus': 'Penn Station (34th St)',
    'direction': 'Northbound',
    'story': 'Model predicts headway bunching during peak commute'
}
```

#### B. Service Disruption Recovery
```python
# Find a period after a delay where headways normalize
scenario = {
    'name': 'Recovery After Delay',
    'time': 'TBD - find from data',
    'story': 'Model captures how service recovers from disruption'
}
```

#### C. Late Night Sparse Service
```python
# 11 PM - 12 AM when headways are naturally longer
scenario = {
    'name': 'Late Night Service',
    'time': '2025-03-15 23:00:00',
    'story': 'Model handles sparse service patterns'
}
```

### Implementation: `src/visualization/narrative.py`

```python
class NarrativeVisualizer:
    """Create story-driven visualizations for specific scenarios."""
    
    def __init__(self, model, data_gen, scaler, config):
        self.model = model
        self.data_gen = data_gen
        self.scaler = scaler
        self.config = config
    
    def find_scenario_sample(self, timestamp, station_name=None):
        """Find the dataset index closest to a given timestamp."""
        pass
    
    def plot_single_station_timeline(self, sample_idx, station_idx, direction):
        """
        Line plot showing:
        - Past 30 min actual headways
        - Next 15 min predicted vs actual
        - Scheduled headway as reference
        """
        pass
    
    def plot_spatial_snapshot(self, sample_idx, timestep, direction):
        """
        Show headway across all stations at a single point in time.
        Useful for identifying where bunching/gaps occur.
        """
        pass
    
    def create_report(self, scenarios_list, save_dir):
        """Generate a full narrative report with multiple scenarios."""
        pass
```

### Station Mapping
Need to map station indices to human-readable names:

```python
STATION_NAMES = {
    0: 'Inwood-207 St (Terminal)',
    10: '168 St',
    20: '125 St',
    30: '59 St-Columbus Circle',
    40: '34 St-Penn Station',
    50: '14 St',
    60: 'Fulton St',
    65: 'Far Rockaway (Terminal)'
}
```

---

## Tomorrow's Session Agenda

### Morning (if needed): Evaluate Training Results
- [ ] Check RMSE and R² from overnight training
- [ ] Determine if optimization is needed

### Task 1: Horizon Analysis (1-2 hours)
- [ ] Create `HorizonAnalyzer` class
- [ ] Add horizon metrics to notebook
- [ ] Generate horizon degradation plot

### Task 2: Narrative Visualizations (1-2 hours)
- [ ] Create station name mapping
- [ ] Find interesting scenarios in data (rush hour, delays, etc.)
- [ ] Build `NarrativeVisualizer` class
- [ ] Generate 3-5 compelling visualizations

### Task 3: Documentation (30 min)
- [ ] Update README with results
- [ ] Create summary of model performance
- [ ] Document any optimization attempts

---

## Files to Create

```
src/
├── experiments/
│   ├── __init__.py
│   └── horizon_analysis.py      # Horizon-specific metrics
├── visualization/
│   ├── __init__.py
│   └── narrative.py             # Scenario-based visualizations
data/
└── station_mapping.json         # Station ID → Name mapping
```

---

## Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| Test RMSE | ≤ 90 seconds | TBD |
| Test R² | ≥ 0.90 | TBD |
| Horizon t+15 RMSE | ≤ 120 seconds | TBD |
| Narrative visualizations | 3+ scenarios | TBD |

---

*Plan created: January 6, 2026*

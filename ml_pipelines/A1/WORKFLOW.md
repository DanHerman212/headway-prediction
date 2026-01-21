# A1 Pipeline: Development and Deployment Workflow

## Key Principles

1. **Never deploy without local testing** - Your integration test should catch 90% of issues
2. **Fail fast with validation** - Validate configs, inputs, outputs at every step
3. **Make debugging easy** - Comprehensive logging, artifact inspection tools
4. **Separate concerns** - Component logic separate from KFP wrappers
5. **Automate checks** - Pre-deployment script catches common mistakes
6. **Document assumptions** - Comment artifact formats, expected shapes, etc.

## Directory Structure

```
A1/
├── src/                    # Core ML logic
│   ├── config.py          # Configuration
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Training logic
│   └── model.py           # Model architecture
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_dataset.py    # Dataset creation tests
│   ├── test_preprocess.py # Preprocessing tests
│   ├── test_model.py      # Model tests
│   └── test_integration.py # Full pipeline simulation
├── utils/                  # Shared utilities
│   ├── artifact_utils.py  # Artifact handling
│   └── logging_utils.py   # Logging helpers
├── debug_tools/            # Debugging utilities
│   ├── inspect_artifact.py # Inspect numpy artifacts
│   └── test_component.py   # Test individual components
├── pipeline.py             # KFP pipeline definition
├── pre_deploy.sh          # Pre-deployment validation
└── deploy.sh              # Deployment script
```

## Development Workflow

### 1. Make Code Changes

Edit files in `src/`:
- `preprocess.py` - Data extraction and preprocessing
- `train.py` - Training logic with Vertex AI integration
- `model.py` - Model architecture
- `config.py` - Hyperparameters and configuration

### 2. Run Unit Tests

Test individual components:

```bash
# Test dataset creation (catches batch size mismatches)
pytest tests/test_dataset.py -v

# Test preprocessing
pytest tests/test_preprocess.py -v

# Test model
pytest tests/test_model.py -v
```

### 3. Run Integration Test

**CRITICAL**: This simulates the full pipeline locally and catches most deployment issues:

```bash
pytest tests/test_integration.py::test_full_pipeline_simulation -v -s
```

This test:
- Saves data to artifact directories (mimics KFP behavior)
- Creates datasets with proper batching
- Trains model for a few epochs
- Validates no shape mismatches occur

**If this test passes, your pipeline is 90% likely to work on Vertex AI.**

### 4. Run Pre-Deployment Validation

Before deploying, run the comprehensive validation script:

```bash
./pre_deploy.sh
```

This checks:
- ✓ Python environment and imports
- ✓ Configuration validity
- ✓ All unit tests pass
- ✓ Integration test passes
- ✓ Common mistakes (drop_remainder, Vertex AI params, artifact handling)
- ✓ Docker and deployment scripts

**Only deploy if all checks pass.**

### 5. Deploy to Vertex AI

```bash
./deploy.sh <run_name>
```

Example:
```bash
./deploy.sh baseline2
```

This will:
1. Build Docker container
2. Push to Container Registry
3. Submit pipeline to Vertex AI
4. Start training job

## Debugging Tools

### Inspect Artifacts

View contents of numpy artifacts (local or GCS):

```bash
python debug_tools/inspect_artifact.py /path/to/artifact_dir
python debug_tools/inspect_artifact.py gs://ml-pipelines-headway-prediction/artifacts/run_123/preprocessed_npy
```

Shows:
- Shape, dtype, size
- Statistical summary (min, max, mean, std)
- Feature-wise breakdown
- Data quality checks (NaN, inf)
- Validation results

### Compare Artifacts

Compare two artifacts to identify differences:

```bash
python debug_tools/inspect_artifact.py /path/to/artifact1 --compare /path/to/artifact2
```

### Test Individual Components

Test components in isolation:

```bash
# Test dataset creation only
python debug_tools/test_component.py --component dataset

# Test preprocessing only
python debug_tools/test_component.py --component preprocess

# Test training with existing data
python debug_tools/test_component.py --component train --preprocessed-data /path/to/data.npy
```

## Common Issues and Solutions

### Issue: "logits_size=[63,3] labels_size=[64,3]"

**Cause**: Batch size mismatch when sample count not divisible by BATCH_SIZE.

**Solution**: Use `drop_remainder=True` in all `.batch()` calls:

```python
target_dataset = tf.data.Dataset.from_tensor_slices({...}).batch(
    config.BATCH_SIZE, 
    drop_remainder=True  # ← CRITICAL
)
```

**Prevention**: Integration test catches this before deployment.

### Issue: "Failed to interpret file as a pickle"

**Cause**: Trying to save/load KFP artifact path as a file (it's a directory).

**Solution**: Save files INSIDE the artifact directory:

```python
# WRONG
np.save(artifact.path, data)  # artifact.path is a DIRECTORY

# CORRECT
output_file = Path(artifact.path) / 'data.npy'
np.save(output_file, data)
```

**Prevention**: Use `utils/artifact_utils.py` helpers:

```python
from utils.artifact_utils import save_to_artifact, load_from_artifact

save_to_artifact(data, artifact.path, 'data.npy')
data = load_from_artifact(artifact.path, 'data.npy')
```

### Issue: "Vertex AI Experiments not available: No experiment set"

**Cause**: Missing `experiment` parameter in `aiplatform.init()`.

**Solution**:

```python
aiplatform.init(
    project=config.BQ_PROJECT,
    location=config.BQ_LOCATION,
    experiment=config.EXPERIMENT_NAME  # ← REQUIRED
)
```

**Prevention**: Pre-deployment script checks for this pattern.

### Issue: No metrics in TensorBoard

**Cause**: TensorBoard callback not properly configured.

**Solution**: Ensure TensorBoard callback points to GCS:

```python
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=f"{config.TENSORBOARD_LOG_DIR}/{run_name}",
        histogram_freq=1,
        profile_batch='10,20'
    )
]
```

**Verification**: Check logs for "Saving TensorBoard logs to gs://..."

## Best Practices

### 1. Artifact Handling

```python
# Always treat artifact.path as a directory
artifact_dir = Path(artifact.path)
output_file = artifact_dir / 'data.npy'
np.save(output_file, data)

# Load similarly
input_file = Path(artifact.path) / 'data.npy'
data = np.load(input_file)
```

### 2. Data Validation

Always validate data after loading:

```python
X = np.load(data_path)

# Check for invalid values
assert not np.isnan(X).any(), "Data contains NaN"
assert not np.isinf(X).any(), "Data contains inf"

# Check expected shape/dtype
assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
assert X.shape[1] == 8, f"Expected 8 features, got {X.shape[1]}"
```

### 3. Dataset Creation

Always use `drop_remainder=True` to prevent batch size mismatches:

```python
dataset = tf.data.Dataset.from_tensor_slices(data).batch(
    batch_size, drop_remainder=True
)
```

### 4. Logging

Use structured logging instead of print:

```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Processing {n_samples} samples")
logger.warning(f"Missing {missing_count} values, filling with 0")
logger.error(f"Validation failed: {error}")
```

### 5. Error Handling

Fail fast with clear error messages:

```python
if data.shape[1] != expected_features:
    raise ValueError(
        f"Feature mismatch: expected {expected_features}, got {data.shape[1]}. "
        f"Check preprocessing configuration."
    )
```

## Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies (BigQuery, GCS)
- Fast execution (< 1 second each)

### Integration Tests
- Test full workflow with real TensorFlow
- Use temporary directories for artifacts
- Simulate KFP component boundaries
- Should complete in < 30 seconds

### Pre-Deployment Checks
- Run all tests
- Validate configuration
- Check for common mistakes
- Verify Docker setup

## Monitoring

### During Training

Watch for:
1. Metrics in TensorBoard: `gs://ml-pipelines-headway-prediction/tensorboard/a1/`
2. Vertex AI Experiments: Console > Vertex AI > Experiments
3. Pipeline logs: Console > Vertex AI > Pipelines > [run] > Logs

### Key Metrics

- Loss decreasing steadily
- Validation loss not diverging from training loss
- Accuracy improving (target ~60-70% for route classification)
- No NaN/inf in metrics

## Troubleshooting

1. **Pipeline fails immediately**: Check Docker build and image push
2. **Preprocessing fails**: Check BigQuery permissions and query
3. **Training fails at start**: Check data loading and validation
4. **Training fails during epoch**: Check batch consistency, shape mismatches
5. **No experiment tracking**: Check Vertex AI Experiments initialization

## Quick Reference

```bash
# Full development cycle
pytest tests/ -v                    # Run all tests
./pre_deploy.sh                     # Validate before deploy
./deploy.sh exp01                   # Deploy

# Debugging
python debug_tools/inspect_artifact.py <path>  # Inspect data
python debug_tools/test_component.py --component dataset  # Test component

# Common fixes
# Fix batch mismatch: Add drop_remainder=True
# Fix artifact error: Use Path(artifact.path) / 'file.npy'
# Fix no tracking: Add experiment= to aiplatform.init()
```

## Success Criteria

Before considering the pipeline "production-ready":

- [ ] All unit tests pass
- [ ] Integration test passes consistently
- [ ] Pre-deployment validation passes
- [ ] At least one successful full training run on Vertex AI
- [ ] Metrics visible in TensorBoard
- [ ] Experiments tracked in Vertex AI
- [ ] Model artifacts saved correctly
- [ ] Can reproduce results with same run_name

## Next Steps

1. Add more comprehensive tests for edge cases
2. Add model evaluation metrics
3. Add model serving/prediction component
4. Set up CI/CD pipeline
5. Add data drift detection
6. Add model performance monitoring

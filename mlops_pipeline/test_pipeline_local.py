"""
Local integration test for the headway training pipeline.
Tests each step's logic without ZenML orchestration or GPU training.

Usage:
    python mlops_pipeline/test_pipeline_local.py
"""

import sys
import os
import time
import torch
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "local_artifacts/processed_data/training_data.parquet"

def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def test_config_loading():
    """Step 1: Verify Hydra config loads and merges correctly."""
    header("Step 1: Config Loading")
    
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(config_name="config")

    # Validate expected keys exist
    assert "model" in cfg, "Missing 'model' config group"
    assert "processing" in cfg, "Missing 'processing' config group"
    assert "training" in cfg, "Missing 'training' config group"
    assert cfg.model.hidden_size > 0, "hidden_size must be positive"
    assert cfg.model.learning_rate > 0, "learning_rate must be positive"
    assert len(cfg.model.quantiles) == 3, f"Expected 3 quantiles, got {len(cfg.model.quantiles)}"
    assert cfg.processing.target == "service_headway", f"Unexpected target: {cfg.processing.target}"

    print(f"  ✅ Config loaded: {len(OmegaConf.to_container(cfg, resolve=True))} top-level keys")
    print(f"     model.hidden_size={cfg.model.hidden_size}, lr={cfg.model.learning_rate}")
    print(f"     processing.max_encoder_length={cfg.processing.max_encoder_length}")
    print(f"     training.max_epochs={cfg.training.max_epochs}, batch_size={cfg.training.batch_size}")
    return cfg


def test_config_with_overrides():
    """Step 1b: Verify Hydra overrides work (simulates CLI overrides on Vertex)."""
    header("Step 1b: Config Overrides")

    import hydra
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["training.max_epochs=5", "model.hidden_size=64"]
        )

    assert cfg.training.max_epochs == 5, f"Override failed: max_epochs={cfg.training.max_epochs}"
    assert cfg.model.hidden_size == 64, f"Override failed: hidden_size={cfg.model.hidden_size}"

    print(f"  ✅ Overrides work: max_epochs=5, hidden_size=64")
    return cfg


def test_data_ingestion():
    """Step 2: Verify parquet ingestion."""
    header("Step 2: Data Ingestion")

    assert os.path.exists(DATA_PATH), f"Test data not found at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)

    assert len(df) > 0, "Dataframe is empty"
    assert "service_headway" in df.columns, "Missing target column 'service_headway'"
    assert "arrival_time" in df.columns, "Missing 'arrival_time' column"
    assert "group_id" in df.columns, "Missing 'group_id' column"

    print(f"  ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"     Groups: {df['group_id'].unique().tolist()}")
    print(f"     Date range: {df['arrival_time'].min()} → {df['arrival_time'].max()}")
    return df


def test_data_processing(df, cfg):
    """Step 3: Verify cleaning + TimeSeriesDataSet creation."""
    header("Step 3: Data Processing")

    from mlops_pipeline.src.data_processing import clean_dataset, create_datasets

    t0 = time.time()
    cleaned = clean_dataset(df)
    t_clean = time.time() - t0

    # Validate cleaning
    assert "time_idx" in cleaned.columns, "clean_dataset didn't create 'time_idx'"
    assert "stops_at_23rd" in cleaned.columns, "clean_dataset didn't create 'stops_at_23rd'"
    assert "arrival_time_dt" in cleaned.columns, "clean_dataset didn't create 'arrival_time_dt'"
    assert cleaned["service_headway"].isna().sum() == 0, "Target has NaN values after cleaning"

    print(f"  ✅ Cleaning passed ({t_clean:.1f}s): {len(cleaned):,} rows")

    # Validate column lists from config match actual columns
    proc = cfg.processing
    for col_list_name in [
        "static_categoricals", "time_varying_known_categoricals",
        "time_varying_known_reals", "time_varying_unknown_categoricals",
        "time_varying_unknown_reals"
    ]:
        for col in list(getattr(proc, col_list_name)):
            assert col in cleaned.columns, f"Config references column '{col}' ({col_list_name}) but it's missing from cleaned data"
    
    print(f"  ✅ All config column references match cleaned dataframe")

    t0 = time.time()
    training, validation, test = create_datasets(cleaned, proc)
    t_ds = time.time() - t0

    print(f"  ✅ TimeSeriesDataSet creation passed ({t_ds:.1f}s)")
    print(f"     Training:   {len(training):,} samples")
    print(f"     Validation: {len(validation):,} samples")
    print(f"     Test:       {len(test):,} samples")

    assert len(training) > 0, "Training dataset is empty"
    assert len(validation) > 0, "Validation dataset is empty"
    assert len(test) > 0, "Test dataset is empty"

    return training, validation, test


def test_model_creation(training, cfg):
    """Step 4: Verify TFT model initializes correctly from dataset + config."""
    header("Step 4: Model Creation")

    from mlops_pipeline.src.model_definitions import create_model

    tft = create_model(training, cfg)

    param_count = sum(p.numel() for p in tft.parameters())
    trainable = sum(p.numel() for p in tft.parameters() if p.requires_grad)

    print(f"  ✅ Model created: {type(tft).__name__}")
    print(f"     Total params:     {param_count:,}")
    print(f"     Trainable params: {trainable:,}")
    print(f"     Output size:      {tft.hparams.output_size}")
    print(f"     Hidden size:      {tft.hparams.hidden_size}")

    assert param_count > 0, "Model has no parameters"
    assert trainable > 0, "Model has no trainable parameters"

    return tft


def test_forward_pass(tft, training):
    """Step 5: Single forward pass — validates tensor shapes and dtypes."""
    header("Step 5: Forward Pass (1 batch)")

    dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
    batch = next(iter(dataloader))

    x, y = batch
    print(f"  Input batch keys: {list(x.keys()) if isinstance(x, dict) else type(x)}")
    print(f"  Target shape: {y[0].shape if isinstance(y, (list, tuple)) else y.shape}")

    tft.eval()
    with torch.no_grad():
        t0 = time.time()
        output = tft(x)
        t_fwd = time.time() - t0

    pred = output.prediction
    print(f"  ✅ Forward pass succeeded ({t_fwd:.2f}s)")
    print(f"     Prediction shape: {pred.shape}")
    print(f"     Expected: (batch, {cfg.processing.max_prediction_length}, {len(list(cfg.model.quantiles))})")
    
    assert pred.shape[0] <= 32, f"Batch dim wrong: {pred.shape[0]}"
    assert pred.shape[2] == len(list(cfg.model.quantiles)), \
        f"Quantile dim mismatch: {pred.shape[2]} vs {len(list(cfg.model.quantiles))}"
    assert not torch.isnan(pred).any(), "Predictions contain NaN!"
    assert not torch.isinf(pred).any(), "Predictions contain Inf!"


def test_dataloader_for_training(training, validation, cfg):
    """Step 6: Verify train/val dataloaders work with configured batch sizes."""
    header("Step 6: DataLoader Validation")

    train_dl = training.to_dataloader(
        train=True,
        batch_size=cfg.training.batch_size,
        num_workers=0
    )
    val_dl = validation.to_dataloader(
        train=False,
        batch_size=cfg.training.batch_size * cfg.training.val_batch_size_multiplier,
        num_workers=0
    )

    train_batch = next(iter(train_dl))
    val_batch = next(iter(val_dl))

    print(f"  ✅ Train dataloader: batch_size={cfg.training.batch_size}, got {train_batch[1][0].shape[0]} samples")
    print(f"  ✅ Val dataloader:   batch_size={cfg.training.batch_size * cfg.training.val_batch_size_multiplier}, got {val_batch[1][0].shape[0]} samples")


def test_evaluation_metrics(tft, test_ds, cfg):
    """Step 7: Verify evaluation metrics (MAE, SMAPE) compute correctly."""
    header("Step 7: Evaluation Metrics")

    from pytorch_forecasting.metrics import MAE, SMAPE

    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=cfg.training.batch_size * cfg.training.val_batch_size_multiplier,
        num_workers=0
    )

    tft.eval()
    t0 = time.time()
    raw_prediction = tft.predict(test_loader, mode="raw", return_x=True)
    t_pred = time.time() - t0

    predictions = raw_prediction.output["prediction"]
    x = raw_prediction.x
    actuals = x["decoder_target"].cpu()
    predictions_cpu = predictions.cpu()

    print(f"  Prediction time: {t_pred:.1f}s")
    print(f"  predictions shape: {predictions_cpu.shape}")
    print(f"  actuals shape:     {actuals.shape}")

    # Validate shapes match
    assert predictions_cpu.shape[0] == actuals.shape[0], \
        f"Batch mismatch: predictions {predictions_cpu.shape[0]} vs actuals {actuals.shape[0]}"
    assert predictions_cpu.shape[1] == actuals.shape[1], \
        f"Seq len mismatch: predictions {predictions_cpu.shape[1]} vs actuals {actuals.shape[1]}"

    # P50 (Median) for point metrics
    p50_forecast = predictions_cpu[:, :, 1]

    mae_metric = MAE()
    smape_metric = SMAPE()

    mae = mae_metric(p50_forecast, actuals).item()
    smape = smape_metric(p50_forecast, actuals).item()

    assert not pd.isna(mae), "MAE is NaN!"
    assert not pd.isna(smape), "SMAPE is NaN!"
    assert mae >= 0, f"MAE should be non-negative, got {mae}"
    assert smape >= 0, f"SMAPE should be non-negative, got {smape}"

    print(f"  ✅ MAE:   {mae:.4f}")
    print(f"  ✅ sMAPE: {smape:.4f}")

    return raw_prediction, predictions, x


def test_group_encoder(tft):
    """Step 8: Verify group encoder is accessible (needed by RushHourVisualizer)."""
    header("Step 8: Group Encoder Access")

    # This is how evaluate_model.py accesses it:
    if hasattr(tft, "dataset_parameters") and "categorical_encoders" in tft.dataset_parameters:
        encoders = tft.dataset_parameters["categorical_encoders"]
        print(f"  Available encoders: {list(encoders.keys())}")

        assert "group_id" in encoders, \
            f"'group_id' not in categorical_encoders. Available: {list(encoders.keys())}"
        
        group_encoder = encoders["group_id"]
        print(f"  group_id encoder type: {type(group_encoder).__name__}")

        # Test inverse transform
        if hasattr(group_encoder, "classes_"):
            print(f"  Encoder classes: {group_encoder.classes_}")
            # Verify we can decode index 0
            decoded = group_encoder.classes_[0]
            print(f"  Index 0 → '{decoded}'")
        elif hasattr(group_encoder, "inverse_transform"):
            import numpy as np
            decoded = group_encoder.inverse_transform(np.array([0]))
            print(f"  Index 0 → '{decoded}'")
        else:
            print(f"  ⚠️  Encoder has no classes_ or inverse_transform — visualizer will use fallback")
    else:
        print(f"  ⚠️  dataset_parameters or categorical_encoders not found")
        print(f"     hasattr(dataset_parameters): {hasattr(tft, 'dataset_parameters')}")
        if hasattr(tft, "dataset_parameters"):
            print(f"     dataset_parameters keys: {list(tft.dataset_parameters.keys()) if isinstance(tft.dataset_parameters, dict) else type(tft.dataset_parameters)}")

    print(f"  ✅ Group encoder check complete")


def test_rush_hour_visualizer(predictions, x, tft):
    """Step 9: Verify RushHourVisualizer produces a valid figure."""
    header("Step 9: Rush Hour Visualization")

    from mlops_pipeline.src.steps.evaluate_model import RushHourVisualizer
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Get group encoder (same logic as evaluate_model step)
    group_encoder = None
    if hasattr(tft, "dataset_parameters") and "categorical_encoders" in tft.dataset_parameters:
        group_encoder = tft.dataset_parameters["categorical_encoders"].get("group_id")

    viz = RushHourVisualizer(predictions, x, group_encoder)

    # Test internal dataframe reconstruction
    df = viz._reconstruct_dataframe()
    print(f"  Reconstructed DF shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Groups found: {df['group'].unique().tolist()}")
    print(f"  Actual range:  [{df['actual'].min():.2f}, {df['actual'].max():.2f}]")
    print(f"  Pred P50 range: [{df['pred_p50'].min():.2f}, {df['pred_p50'].max():.2f}]")

    assert len(df) > 0, "Reconstructed dataframe is empty"
    assert not df["actual"].isna().any(), "Actuals contain NaN"
    assert not df["pred_p50"].isna().any(), "P50 predictions contain NaN"

    # Test plot generation
    fig = viz.plot_rush_hour(window_size=180)
    assert fig is not None, "plot_rush_hour returned None"

    n_axes = len(fig.get_axes())
    print(f"  ✅ Figure generated with {n_axes} subplot(s)")
    
    plt.close(fig)
    print(f"  ✅ Visualization test passed")


if __name__ == "__main__":
    print("🚀 Headway Pipeline — Local Integration Test")
    print(f"   Data: {DATA_PATH}")
    
    t_start = time.time()
    failures = []

    try:
        cfg = test_config_loading()
    except Exception as e:
        failures.append(("Config Loading", e))
        print(f"  ❌ FAILED: {e}")
        sys.exit(1)  # Can't continue without config

    try:
        test_config_with_overrides()
    except Exception as e:
        failures.append(("Config Overrides", e))
        print(f"  ❌ FAILED: {e}")

    try:
        df = test_data_ingestion()
    except Exception as e:
        failures.append(("Data Ingestion", e))
        print(f"  ❌ FAILED: {e}")
        sys.exit(1)  # Can't continue without data

    try:
        training, validation, test = test_data_processing(df, cfg)
    except Exception as e:
        failures.append(("Data Processing", e))
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        tft = test_model_creation(training, cfg)
    except Exception as e:
        failures.append(("Model Creation", e))
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        test_forward_pass(tft, training)
    except Exception as e:
        failures.append(("Forward Pass", e))
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        test_dataloader_for_training(training, validation, cfg)
    except Exception as e:
        failures.append(("DataLoader Validation", e))
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        raw_prediction, predictions, x = test_evaluation_metrics(tft, test, cfg)
    except Exception as e:
        failures.append(("Evaluation Metrics", e))
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()
        raw_prediction, predictions, x = None, None, None

    try:
        test_group_encoder(tft)
    except Exception as e:
        failures.append(("Group Encoder Access", e))
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()

    if predictions is not None and x is not None:
        try:
            test_rush_hour_visualizer(predictions, x, tft)
        except Exception as e:
            failures.append(("Rush Hour Visualization", e))
            print(f"  ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
    else:
        print("\n  ⏭️  Skipping visualization test (no predictions from metrics step)")

    elapsed = time.time() - t_start

    header("RESULTS")
    total_tests = 10
    passed = total_tests - len(failures)
    if not failures:
        print(f"  🎉 ALL {total_tests} TESTS PASSED in {elapsed:.1f}s")
        print(f"  Pipeline logic is sound — safe to deploy to Vertex AI.")
    else:
        print(f"  ⚠️  {passed}/{total_tests} passed, {len(failures)} FAILED in {elapsed:.1f}s:")
        for name, err in failures:
            print(f"     ❌ {name}: {err}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Smoke Test for Pipeline Components

Validates that all pipeline components work together locally before
deploying to Vertex AI. Runs a minimal training loop (2 epochs, small batch)
to verify:
    1. Config serialization/deserialization
    2. Data loading and MinMaxScaler fitting
    3. Model building via HeadwayConvLSTM
    4. Training via Trainer
    5. Evaluation metrics computation

Usage:
    # In container
    docker run --rm -v $(pwd)/data:/app/data headway-training:local \
        -m src.experiments.smoke_test

    # Local (with venv)
    python -m src.experiments.smoke_test
"""

import json
import os
import sys
import tempfile
import numpy as np


def run_smoke_test():
    """Run minimal end-to-end test of all pipeline components."""
    
    print("=" * 60)
    print("SMOKE TEST: Pipeline Component Validation")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Config Serialization
    # =========================================================================
    print("\n[1/5] Testing Config serialization...")
    
    from src.config import Config
    
    config = Config()
    
    # Override for fast smoke test
    config.EPOCHS = 2
    config.BATCH_SIZE = 8
    config.EARLY_STOPPING_PATIENCE = 2
    
    # Serialize to JSON (as pipeline does)
    config_dict = {
        k: v for k, v in vars(config).items() 
        if not k.startswith('_') and not callable(v)
    }
    config_dict["KERNEL_SIZE"] = list(config.KERNEL_SIZE)
    config_json = json.dumps(config_dict)
    
    # Deserialize (as components do)
    restored_dict = json.loads(config_json)
    restored_config = Config(**{k: v for k, v in restored_dict.items() 
                                 if hasattr(Config, k) and not k.startswith('_')})
    
    assert restored_config.EPOCHS == 2, "Config restore failed"
    assert restored_config.TRAIN_SPLIT == 0.6, "Split ratios not in Config"
    print(f"   ✓ Config serialized/deserialized: EPOCHS={restored_config.EPOCHS}, SPLITS={restored_config.TRAIN_SPLIT}/{restored_config.VAL_SPLIT}/{restored_config.TEST_SPLIT}")
    
    # =========================================================================
    # Step 2: Data Loading + Scaling
    # =========================================================================
    print("\n[2/5] Testing data loading and MinMaxScaler...")
    
    from sklearn.preprocessing import MinMaxScaler
    
    # Check data files exist
    assert os.path.exists(config.headway_path), f"Headway file not found: {config.headway_path}"
    assert os.path.exists(config.schedule_path), f"Schedule file not found: {config.schedule_path}"
    
    # Load raw data
    headway_data = np.load(config.headway_path).astype('float32')
    schedule_data = np.load(config.schedule_path).astype('float32')
    
    print(f"   Headway shape: {headway_data.shape}")
    print(f"   Schedule shape: {schedule_data.shape}")
    
    # Calculate splits
    total_window = config.LOOKBACK_MINS + config.FORECAST_MINS
    total_samples = len(headway_data) - total_window
    train_end = int(total_samples * config.TRAIN_SPLIT)
    val_end = int(total_samples * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    
    print(f"   Split indices: train=[0:{train_end}], val=[{train_end}:{val_end}], test=[{val_end}:{total_samples}]")
    
    # Fit scaler on training data ONLY
    train_headway = headway_data[:train_end + total_window]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_headway.reshape(-1, 1))
    
    scaler_params = {
        "data_min": float(scaler.data_min_[0]),
        "data_max": float(scaler.data_max_[0]),
    }
    print(f"   ✓ Scaler fitted on training data: min={scaler_params['data_min']:.2f}, max={scaler_params['data_max']:.2f}")
    
    # Transform all data
    headway_scaled = scaler.transform(headway_data.reshape(-1, 1)).reshape(headway_data.shape)
    schedule_scaled = scaler.transform(schedule_data.reshape(-1, 1)).reshape(schedule_data.shape)
    headway_scaled = np.clip(headway_scaled, 0, 1)
    schedule_scaled = np.clip(schedule_scaled, 0, 1)
    
    # Save to temp dir
    temp_dir = tempfile.mkdtemp()
    np.save(os.path.join(temp_dir, "headway_scaled.npy"), headway_scaled)
    np.save(os.path.join(temp_dir, "schedule_scaled.npy"), schedule_scaled)
    print(f"   ✓ Scaled data saved to temp: {temp_dir}")
    
    # =========================================================================
    # Step 3: Dataset Creation
    # =========================================================================
    print("\n[3/5] Testing SubwayDataGenerator...")
    
    from src.data.dataset import SubwayDataGenerator
    
    # Point config to scaled data
    config.DATA_DIR = temp_dir
    config.HEADWAY_FILE = "headway_scaled.npy"
    config.SCHEDULE_FILE = "schedule_scaled.npy"
    
    gen = SubwayDataGenerator(config)
    gen.load_data(normalize=False)  # Already scaled
    
    # Use small subset for smoke test
    smoke_train_end = min(500, train_end)
    smoke_val_end = min(600, val_end)
    
    train_ds = gen.make_dataset(start_index=0, end_index=smoke_train_end, shuffle=True)
    val_ds = gen.make_dataset(start_index=smoke_train_end, end_index=smoke_val_end, shuffle=False)
    
    # Verify shapes
    for batch_x, batch_y in train_ds.take(1):
        headway_input = batch_x["headway_input"]
        schedule_input = batch_x["schedule_input"]
        print(f"   Batch shapes:")
        print(f"     headway_input: {headway_input.shape}")
        print(f"     schedule_input: {schedule_input.shape}")
        print(f"     target: {batch_y.shape}")
    
    print(f"   ✓ Datasets created successfully")
    
    # =========================================================================
    # Step 4: Model + Training
    # =========================================================================
    print("\n[4/5] Testing HeadwayConvLSTM + Trainer...")
    
    from src.models.baseline_convlstm import HeadwayConvLSTM
    from src.training.trainer import Trainer
    
    # Build model
    model_builder = HeadwayConvLSTM(config)
    model = model_builder.build_model()
    print(f"   Model built: {model.count_params():,} parameters")
    
    # Train (2 epochs)
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")
    trainer = Trainer(model, config, checkpoint_dir=checkpoint_dir)
    trainer.compile_model()
    
    print(f"   Running {config.EPOCHS} epochs (smoke test)...")
    history = trainer.fit(train_ds, val_ds)
    
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    print(f"   ✓ Training complete: loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")
    
    # =========================================================================
    # Step 5: Evaluation
    # =========================================================================
    print("\n[5/5] Testing evaluation metrics...")
    
    # Get predictions
    test_ds = gen.make_dataset(start_index=smoke_val_end, end_index=smoke_val_end + 100, shuffle=False)
    
    all_preds = []
    all_targets = []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        all_preds.append(preds)
        all_targets.append(batch_y.numpy())
    
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform
        preds_original = scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
        targets_original = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
        
        # RMSE in seconds
        rmse_minutes = np.sqrt(np.mean((preds_original - targets_original) ** 2))
        rmse_seconds = rmse_minutes * 60
        
        # R-squared
        ss_res = np.sum((targets_original - preds_original) ** 2)
        ss_tot = np.sum((targets_original - np.mean(targets_original)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"   ✓ Test metrics: RMSE={rmse_seconds:.2f}s, R²={r_squared:.4f}")
    else:
        print(f"   ⚠ No test batches (dataset too small)")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("✓ SMOKE TEST PASSED - All components wired correctly")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
KFP Pipeline for Headway Prediction

Orchestrates the existing modules as pipeline steps:
    Step 1: dataset.py    - Data loading, splitting, and scaling
    Step 2: trainer.py    - Model training with tracking
    Step 3: evaluator.py  - Model evaluation on test set

All hyperparameters flow from src/config.py.
Scaling is fit on training data only, then applied to val/test.

Usage:
    # Compile pipeline
    python -m src.experiments.pipeline --compile

    # Submit to Vertex AI
    python -m src.experiments.pipeline --submit --run_name baseline-001
"""

import argparse
import json
from datetime import datetime
from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform


# =============================================================================
# GCP Configuration
# =============================================================================

PROJECT = "time-series-478616"
REGION = "us-east1"
BUCKET = "st-convnet-training-configuration"
PIPELINE_ROOT = f"gs://{BUCKET}/pipeline-runs"
TENSORBOARD_LOG_ROOT = f"gs://{BUCKET}/tensorboard"
SERVICE_ACCOUNT = f"vertex-ai-sa@{PROJECT}.iam.gserviceaccount.com"

# Container with all dependencies (Artifact Registry)
TRAINING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT}/headway-prediction/training:latest"


# =============================================================================
# Step 1: Data Component
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def data_component(
    config_json: str,
    run_name: str,
    data_info_output: dsl.Output[dsl.Artifact],
):
    """
    Load and prepare data using SubwayDataGenerator with proper scaling.
    
    Scaling approach:
        1. Load raw data (no normalization)
        2. Split into train/val/test using 60/20/20 from Config
        3. Fit MinMaxScaler on training data ONLY
        4. Transform all splits with training scaler
        5. Output data info + scaler params for downstream components
    
    Args:
        config_json: Serialized Config object as JSON
    
    Returns:
        Path to JSON file with data paths, shapes, and scaler params
    """
    import json
    import os
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from src.config import Config
    
    # Deserialize config
    config_dict = json.loads(config_json)
    config = Config(**{k: v for k, v in config_dict.items() 
                       if hasattr(Config, k) and not k.startswith('_')})
    
    # Handle GCS path
    if config.DATA_GCS_PATH.startswith("gs://"):
        from google.cloud import storage
        import tempfile
        
        local_dir = tempfile.mkdtemp()
        path = config.DATA_GCS_PATH.replace("gs://", "")
        bucket_name = path.split("/")[0]
        prefix = "/".join(path.split("/")[1:])
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        for filename in [config.HEADWAY_FILE, config.SCHEDULE_FILE]:
            blob_path = f"{prefix}/{filename}" if prefix else filename
            blob = bucket.blob(blob_path)
            local_path = os.path.join(local_dir, filename)
            blob.download_to_filename(local_path)
            print(f"Downloaded: {blob_path} -> {local_path}")
        
        config.DATA_DIR = local_dir
    
    # Load raw data (NO normalization - we handle scaling explicitly)
    print(f"Loading data from {config.DATA_DIR}...")
    headway_data = np.load(config.headway_path).astype('float32')
    schedule_data = np.load(config.schedule_path).astype('float32')
    
    print(f"Headway shape: {headway_data.shape}")
    print(f"Schedule shape: {schedule_data.shape}")
    
    # Calculate split indices using Config ratios
    total_window = config.LOOKBACK_MINS + config.FORECAST_MINS
    total_samples = len(headway_data) - total_window
    
    train_end = int(total_samples * config.TRAIN_SPLIT)
    val_end = int(total_samples * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    
    print(f"Split indices: train=[0:{train_end}], val=[{train_end}:{val_end}], test=[{val_end}:{total_samples}]")
    print(f"Split sizes: train={train_end}, val={val_end - train_end}, test={total_samples - val_end}")
    
    # Fit MinMaxScaler on TRAINING data only
    # Flatten for sklearn, then reshape back
    train_headway = headway_data[:train_end + total_window]
    original_shape = train_headway.shape
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_flat = train_headway.reshape(-1, 1)
    scaler.fit(train_flat)
    
    # Store scaler parameters for reconstruction in downstream components
    scaler_params = {
        "data_min": float(scaler.data_min_[0]),
        "data_max": float(scaler.data_max_[0]),
        "scale": float(scaler.scale_[0]),
        "min_": float(scaler.min_[0]),
    }
    print(f"Scaler fitted on training data: min={scaler_params['data_min']:.2f}, max={scaler_params['data_max']:.2f}")
    
    # Transform ALL data using training scaler
    headway_scaled = scaler.transform(headway_data.reshape(-1, 1)).reshape(headway_data.shape)
    schedule_scaled = scaler.transform(schedule_data.reshape(-1, 1)).reshape(schedule_data.shape)
    
    # Clip to [0, 1] - handles values outside training range
    headway_scaled = np.clip(headway_scaled, 0, 1)
    schedule_scaled = np.clip(schedule_scaled, 0, 1)
    
    # Save scaled data locally first
    scaled_dir = os.path.join(config.DATA_DIR, "scaled")
    os.makedirs(scaled_dir, exist_ok=True)
    
    headway_path = os.path.join(scaled_dir, "headway_scaled.npy")
    schedule_path = os.path.join(scaled_dir, "schedule_scaled.npy")
    
    np.save(headway_path, headway_scaled)
    np.save(schedule_path, schedule_scaled)
    print(f"Saved scaled data locally to {scaled_dir}")
    
    # Upload scaled data to GCS so training component can access it
    gcs_scaled_dir = f"gs://{bucket_name}/headway-prediction/scaled/{run_name}"
    for local_file, filename in [(headway_path, "headway_scaled.npy"), (schedule_path, "schedule_scaled.npy")]:
        gcs_blob_path = f"headway-prediction/scaled/{run_name}/{filename}"
        blob = bucket.blob(gcs_blob_path)
        blob.upload_from_filename(local_file)
        print(f"Uploaded: {local_file} -> gs://{bucket_name}/{gcs_blob_path}")
    
    # Output data info for downstream components (with GCS paths)
    data_info = {
        "scaled_data_gcs": gcs_scaled_dir,
        "headway_file": "headway_scaled.npy",
        "schedule_file": "schedule_scaled.npy",
        "total_samples": total_samples,
        "train_end": train_end,
        "val_end": val_end,
        "headway_shape": list(headway_data.shape),
        "schedule_shape": list(schedule_data.shape),
        "scaler_params": scaler_params,
        "config": config_dict,
    }
    
    print(f"Data preparation complete: {json.dumps(data_info, indent=2)}")
    
    with open(data_info_output.path, "w") as f:
        json.dump(data_info, f)


# =============================================================================
# Step 2: Training Component
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def training_component(
    data_info: dsl.Input[dsl.Artifact],
    tensorboard_log_dir: str,
    model_output_dir: str,
    training_info_output: dsl.Output[dsl.Artifact],
):
    """
    Train model using Trainer with tracking integration.
    
    All hyperparameters come from Config (passed through data_info).
    
    Uses:
        - HeadwayConvLSTM for model architecture (uncompiled)
        - Trainer for compilation and training loop
        - Tracker for TensorBoard logging
    """
    import json
    import os
    import tempfile
    import numpy as np
    import tensorflow as tf
    from google.cloud import storage
    from src.config import Config
    from src.models.baseline_convlstm import HeadwayConvLSTM
    from src.training.trainer import Trainer
    from src.tracking import Tracker, TrackerConfig
    
    # Load data info from artifact
    with open(data_info.path, "r") as f:
        data_info_dict = json.load(f)
    
    # Download scaled data from GCS
    gcs_scaled_dir = data_info_dict["scaled_data_gcs"]
    local_dir = tempfile.mkdtemp()
    
    path = gcs_scaled_dir.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for filename in [data_info_dict["headway_file"], data_info_dict["schedule_file"]]:
        blob_path = f"{prefix}/{filename}"
        blob = bucket.blob(blob_path)
        local_path = os.path.join(local_dir, filename)
        blob.download_to_filename(local_path)
        print(f"Downloaded: gs://{bucket_name}/{blob_path} -> {local_path}")
    
    # Reconstruct Config from data_info
    config_dict = data_info_dict["config"]
    config = Config(**{k: v for k, v in config_dict.items() 
                       if hasattr(Config, k) and not k.startswith('_')})
    
    # Override DATA_DIR with local scaled data location
    config.DATA_DIR = local_dir
    config.HEADWAY_FILE = data_info_dict["headway_file"]
    config.SCHEDULE_FILE = data_info_dict["schedule_file"]
    
    # Load scaled data
    from src.data.dataset import SubwayDataGenerator
    gen = SubwayDataGenerator(config)
    gen.load_data(normalize=False)  # Already scaled!
    
    # Create datasets using split indices from data_info
    train_ds = gen.make_dataset(
        start_index=0,
        end_index=data_info_dict["train_end"],
        shuffle=True
    )
    val_ds = gen.make_dataset(
        start_index=data_info_dict["train_end"],
        end_index=data_info_dict["val_end"],
        shuffle=False
    )
    
    # Build model (uncompiled - Trainer handles compilation)
    model_builder = HeadwayConvLSTM(config)
    model = model_builder.build_model()
    
    # Initialize Vertex AI for experiment tracking
    from google.cloud import aiplatform
    aiplatform.init(
        project="time-series-478616",
        location="us-east1",
    )
    
    # Setup tracking with Vertex AI Experiments integration
    run_name = os.path.basename(tensorboard_log_dir)
    tracker_config = TrackerConfig(
        experiment_name="headway-prediction",
        run_name=run_name,
        log_dir=tensorboard_log_dir,
        scalars=True,
        histograms=True,
        histogram_freq=5,
        graphs=True,
        hparams=True,
        hparams_dict={
            "filters": config.FILTERS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "lookback": config.LOOKBACK_MINS,
            "forecast": config.FORECAST_MINS,
            "train_split": config.TRAIN_SPLIT,
            "val_split": config.VAL_SPLIT,
        }
    )
    tracker = Tracker(tracker_config, use_vertex_experiments=True)
    
    # Train using Trainer (handles compilation with Config params)
    trainer = Trainer(model, config, checkpoint_dir=model_output_dir)
    trainer.compile_model()
    
    history = trainer.fit(
        train_ds,
        val_ds,
        extra_callbacks=tracker.keras_callbacks()
    )
    
    # Log final metrics
    tracker.log_scalar("final/val_loss", min(history.history["val_loss"]), step=0)
    if "val_rmse_seconds" in history.history:
        tracker.log_scalar("final/val_rmse_seconds", min(history.history["val_rmse_seconds"]), step=0)
    if "val_r_squared" in history.history:
        tracker.log_scalar("final/val_r_squared", max(history.history["val_r_squared"]), step=0)
    
    tracker.close()
    
    # Output training info
    training_info_data = {
        "model_path": os.path.join(model_output_dir, "best_model.keras"),
        "tensorboard_log_dir": tensorboard_log_dir,
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "data_info": data_info_dict,  # Pass through for evaluation
    }
    
    print(f"Training complete: {json.dumps(training_info_data, indent=2)}")
    
    with open(training_info_output.path, "w") as f:
        json.dump(training_info_data, f)


# =============================================================================
# Step 3: Evaluation Component
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def evaluation_component(
    training_info: dsl.Input[dsl.Artifact],
    tensorboard_log_dir: str,
    eval_info_output: dsl.Output[dsl.Artifact],
):
    """
    Evaluate model on TEST set using Evaluator.
    
    Uses held-out test set (last 20% of data) for final evaluation.
    Scaler params from training are used for inverse transform.
    
    Generates:
        - Test metrics (RMSE in seconds, R-squared)
        - Spatiotemporal visualizations (logged to TensorBoard)
    """
    import json
    import os
    import tempfile
    import numpy as np
    import tensorflow as tf
    from google.cloud import aiplatform, storage
    from sklearn.preprocessing import MinMaxScaler
    from src.config import Config
    from src.data.dataset import SubwayDataGenerator
    from src.evaluator import Evaluator
    from src.tracking import Tracker, TrackerConfig
    
    # Load training info (includes data_info) from artifact
    with open(training_info.path, "r") as f:
        training_info_dict = json.load(f)
    
    data_info_dict = training_info_dict["data_info"]
    
    # Download scaled data from GCS
    gcs_scaled_dir = data_info_dict["scaled_data_gcs"]
    local_dir = tempfile.mkdtemp()
    
    path = gcs_scaled_dir.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for filename in [data_info_dict["headway_file"], data_info_dict["schedule_file"]]:
        blob_path = f"{prefix}/{filename}"
        blob = bucket.blob(blob_path)
        local_path = os.path.join(local_dir, filename)
        blob.download_to_filename(local_path)
        print(f"Downloaded: gs://{bucket_name}/{blob_path} -> {local_path}")
    
    # Reconstruct Config
    config_dict = data_info_dict["config"]
    config = Config(**{k: v for k, v in config_dict.items() 
                       if hasattr(Config, k) and not k.startswith('_')})
    
    # Point to local scaled data
    config.DATA_DIR = local_dir
    config.HEADWAY_FILE = data_info_dict["headway_file"]
    config.SCHEDULE_FILE = data_info_dict["schedule_file"]
    
    # Load model from GCS
    model_gcs_path = training_info_dict["model_path"]
    local_model_path = os.path.join(tempfile.mkdtemp(), "best_model.keras")
    
    model_path = model_gcs_path.replace("gs://", "")
    model_bucket_name = model_path.split("/")[0]
    model_blob_path = "/".join(model_path.split("/")[1:])
    
    model_bucket = client.bucket(model_bucket_name)
    model_blob = model_bucket.blob(model_blob_path)
    model_blob.download_to_filename(local_model_path)
    print(f"Downloaded model: {model_gcs_path} -> {local_model_path}")
    
    model = tf.keras.models.load_model(local_model_path)
    print(f"Loaded model from {local_model_path}")
    
    # Load scaled data
    gen = SubwayDataGenerator(config)
    gen.load_data(normalize=False)  # Already scaled!
    
    # Create TEST dataset (val_end to end)
    test_ds = gen.make_dataset(
        start_index=data_info_dict["val_end"],
        end_index=None,  # Use all remaining data
        shuffle=False
    )
    
    # Reconstruct scaler for inverse transform
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array([data_info_dict["scaler_params"]["data_min"]])
    scaler.data_max_ = np.array([data_info_dict["scaler_params"]["data_max"]])
    scaler.scale_ = np.array([data_info_dict["scaler_params"]["scale"]])
    scaler.min_ = np.array([data_info_dict["scaler_params"]["min_"]])
    
    # Evaluate on test set
    test_results = model.evaluate(test_ds, return_dict=True)
    print(f"Test set results: {test_results}")
    
    # Initialize Vertex AI for experiment tracking
    aiplatform.init(
        project="time-series-478616",
        location="us-east1",
    )
    
    # Setup tracker for visualization logging with Vertex AI Experiments
    run_name = os.path.basename(tensorboard_log_dir)
    tracker_config = TrackerConfig(
        experiment_name="headway-prediction",
        run_name=run_name,
        log_dir=tensorboard_log_dir,
        scalars=True,
        histograms=False,
        graphs=False,
        hparams=False,
    )
    tracker = Tracker(tracker_config, use_vertex_experiments=True)
    
    # Log test metrics
    for name, value in test_results.items():
        tracker.log_scalar(f"test/{name}", float(value), step=0)
    
    # Compute predictions and inverse-transform for real-world metrics
    all_preds = []
    all_targets = []
    
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        all_preds.append(preds)
        all_targets.append(batch_y.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Inverse transform to original scale (minutes)
    preds_original = scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    targets_original = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
    
    # Calculate RMSE in seconds
    rmse_minutes = np.sqrt(np.mean((preds_original - targets_original) ** 2))
    rmse_seconds = rmse_minutes * 60
    
    # Calculate R-squared
    ss_res = np.sum((targets_original - preds_original) ** 2)
    ss_tot = np.sum((targets_original - np.mean(targets_original)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    tracker.log_scalar("test/rmse_seconds", rmse_seconds, step=0)
    tracker.log_scalar("test/r_squared", r_squared, step=0)
    
    print(f"Test RMSE: {rmse_seconds:.2f} seconds")
    print(f"Test R²: {r_squared:.4f}")
    
    # Generate visualization using Evaluator
    evaluator = Evaluator(config)
    
    # Log sample spatiotemporal comparison
    for batch_x, batch_y in test_ds.take(1):
        preds = model.predict(batch_x, verbose=0)
        
        fig = evaluator.plot_spatiotemporal_comparison(
            batch_y[0].numpy(),
            preds[0],
            title="Test Sample"
        )
        
        if fig is not None:
            tracker.log_figure("test/spatiotemporal", fig, step=0)
    
    tracker.close()
    
    # Output evaluation info
    eval_info = {
        "model_path": training_info_dict["model_path"],
        "test_samples": data_info_dict["total_samples"] - data_info_dict["val_end"],
        "test_loss": float(test_results.get("loss", 0)),
        "test_rmse_seconds": float(rmse_seconds),
        "test_r_squared": float(r_squared),
        "tensorboard_log_dir": tensorboard_log_dir,
    }
    
    print(f"Evaluation complete: {json.dumps(eval_info, indent=2)}")
    
    with open(eval_info_output.path, "w") as f:
        json.dump(eval_info, f)


# =============================================================================
# Pipeline Definition
# =============================================================================

@dsl.pipeline(
    name="headway-prediction-pipeline",
    description="Train and evaluate headway prediction model (Config-driven)"
)
def headway_pipeline(
    run_name: str,
):
    """
    End-to-end pipeline: data → training → evaluation
    
    All hyperparameters come from src/config.py.
    The only runtime parameter is run_name for tracking.
    """
    from src.config import Config
    import json
    
    # Serialize Config to JSON for component passing
    config = Config()
    config_dict = {
        k: v for k, v in vars(config).items() 
        if not k.startswith('_') and not callable(v)
    }
    # Handle tuple serialization
    config_dict["KERNEL_SIZE"] = list(config.KERNEL_SIZE)
    config_json = json.dumps(config_dict)
    
    # Paths derived from run_name
    tensorboard_log_dir = f"{TENSORBOARD_LOG_ROOT}/{run_name}"
    model_output_dir = f"gs://{BUCKET}/models/{run_name}"
    
    # Step 1: Data preparation (scaling fit on train only)
    data_task = data_component(config_json=config_json, run_name=run_name)
    
    # Step 2: Training
    training_task = training_component(
        data_info=data_task.outputs["data_info_output"],
        tensorboard_log_dir=tensorboard_log_dir,
        model_output_dir=model_output_dir,
    )
    training_task.set_accelerator_type("NVIDIA_TESLA_A100")
    training_task.set_accelerator_limit(1)
    training_task.set_cpu_limit("8")
    training_task.set_memory_limit("32G")
    
    # Step 3: Evaluation on held-out test set
    eval_task = evaluation_component(
        training_info=training_task.outputs["training_info_output"],
        tensorboard_log_dir=tensorboard_log_dir,
    )


# =============================================================================
# CLI
# =============================================================================

def compile_pipeline(output_path: str = "headway_pipeline.json"):
    """Compile pipeline to JSON."""
    compiler.Compiler().compile(
        pipeline_func=headway_pipeline,
        package_path=output_path
    )
    print(f"Pipeline compiled: {output_path}")


def submit_pipeline(run_name: str):
    """Submit pipeline to Vertex AI."""
    aiplatform.init(project=PROJECT, location=REGION)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not run_name:
        run_name = f"baseline-{timestamp}"
    
    # Compile
    pipeline_file = f"/tmp/headway_pipeline_{timestamp}.json"
    compile_pipeline(pipeline_file)
    
    # Submit
    job = aiplatform.PipelineJob(
        display_name=f"headway-{run_name}",
        template_path=pipeline_file,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={"run_name": run_name},
    )
    
    print(f"Submitting pipeline: {run_name}")
    job.submit()  # Uses default compute service account
    
    print(f"Pipeline submitted: {job.resource_name}")
    print(f"TensorBoard logs: {TENSORBOARD_LOG_ROOT}/{run_name}")
    
    return job


def main():
    parser = argparse.ArgumentParser(description="Headway prediction pipeline")
    parser.add_argument("--compile", action="store_true", help="Compile pipeline only")
    parser.add_argument("--submit", action="store_true", help="Submit to Vertex AI")
    parser.add_argument("--run_name", type=str, default="", help="Run name for tracking")
    
    args = parser.parse_args()
    
    if args.compile:
        compile_pipeline()
    elif args.submit:
        submit_pipeline(run_name=args.run_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

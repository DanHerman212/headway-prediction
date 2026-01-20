"""
Vertex AI Pipeline for A1 Track Headway Prediction

Orchestrates the training pipeline as a sequence of containerized components:
1. Extract data from BigQuery
2. Preprocess features
3. Train model with TensorBoard tracking
4. Evaluate on test set

Usage:
    # Compile pipeline
    python pipeline.py --compile
    
    # Submit to Vertex AI
    python pipeline.py --submit --run_name baseline_001
"""

import argparse
from datetime import datetime
from kfp import dsl, compiler
from google.cloud import aiplatform
from config import config


# =============================================================================
# Pipeline Configuration
# =============================================================================

PROJECT_ID = config.BQ_PROJECT
REGION = config.BQ_LOCATION
BUCKET = config.GCS_BUCKET
PIPELINE_ROOT = f"gs://{BUCKET}/pipeline-runs"

# Container image (to be built and pushed to Artifact Registry)
# Build with: docker build -t {REGION}-docker.pkg.dev/{PROJECT_ID}/ml-pipelines/a1-training:latest .
TRAINING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-pipelines/a1-training:latest"


# =============================================================================
# Component 1: Data Extraction
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def extract_data_component(
    project_id: str,
    dataset: str,
    table: str,
    raw_data_csv: dsl.Output[dsl.Dataset]
):
    """
    Extract A1 track data from BigQuery and save as CSV artifact.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        raw_data_csv: Output CSV artifact
    """
    from google.cloud import bigquery
    import pandas as pd
    
    print(f"Extracting data from BigQuery...")
    print(f"  Project: {project_id}")
    print(f"  Dataset: {dataset}")
    print(f"  Table: {table}")
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # SQL query
    query = f"""
    SELECT 
        arrival_time,
        headway,
        route_id,
        track,
        time_of_day,
        day_of_week
    FROM `{project_id}.{dataset}.{table}`
    WHERE track = 'A1'
    ORDER BY arrival_time ASC
    """
    
    print(f"Executing query...")
    df = client.query(query).to_dataframe()
    
    print(f"Retrieved {len(df):,} records")
    print(f"Date range: {df['arrival_time'].min()} to {df['arrival_time'].max()}")
    
    # Save to artifact
    df.to_csv(raw_data_csv.path, index=False)
    print(f"Saved to: {raw_data_csv.path}")


# =============================================================================
# Component 2: Preprocessing
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def preprocess_component(
    raw_data_csv: dsl.Input[dsl.Dataset],
    preprocessed_npy: dsl.Output[dsl.Dataset],
    metadata_json: dsl.Output[dsl.Artifact]
):
    """
    Preprocess raw CSV data into model-ready numpy arrays.
    
    Applies:
    - Log transformation for headway
    - One-hot encoding for route_id
    - Cyclical encoding for temporal features
    
    Args:
        raw_data_csv: Input CSV artifact
        preprocessed_npy: Output numpy array artifact
        metadata_json: Output metadata artifact
    """
    import numpy as np
    import pandas as pd
    import json
    from preprocess import (
        log_transform_headway,
        one_hot_encode_route,
        cyclical_encode_temporal,
        create_feature_array
    )
    
    print(f"Loading raw data from: {raw_data_csv.path}")
    df = pd.read_csv(raw_data_csv.path)
    print(f"Loaded {len(df):,} records")
    
    # Apply transformations
    df, headway_stats = log_transform_headway(df)
    df = one_hot_encode_route(df)
    df = cyclical_encode_temporal(df)
    
    # Create feature array
    X = create_feature_array(df)
    
    # Save preprocessed data
    np.save(preprocessed_npy.path, X)
    print(f"Saved preprocessed data: {X.shape}")
    
    # Save metadata
    metadata = {
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'headway_stats': headway_stats,
        'date_range': {
            'start': str(df['arrival_time'].min()),
            'end': str(df['arrival_time'].max())
        }
    }
    
    with open(metadata_json.path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata")


# =============================================================================
# Component 3: Training
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def train_component(
    preprocessed_npy: dsl.Input[dsl.Dataset],
    metadata_json: dsl.Input[dsl.Artifact],
    run_name: str,
    experiment_name: str,
    tensorboard_log_dir: str,
    model_output: dsl.Output[dsl.Model],
    history_json: dsl.Output[dsl.Artifact],
    metrics: dsl.Output[dsl.Metrics]
):
    """
    Train stacked GRU model with TensorBoard and Vertex AI Experiments tracking.
    
    Args:
        preprocessed_npy: Input preprocessed data
        metadata_json: Input metadata
        run_name: Unique run identifier
        experiment_name: Vertex AI experiment name
        tensorboard_log_dir: GCS path for TensorBoard logs
        model_output: Output trained model artifact
        history_json: Output training history artifact
        metrics: Output metrics for Vertex AI
    """
    import numpy as np
    import json
    import tensorflow as tf
    from tensorflow import keras
    from google.cloud import aiplatform
    
    from config import config
    from model import get_model
    from train import calculate_split_indices, create_timeseries_datasets
    
    print(f"Training run: {run_name}")
    
    # Initialize Vertex AI Experiments
    aiplatform.init(project=config.BQ_PROJECT, location=config.BQ_LOCATION)
    experiment = aiplatform.Experiment.get_or_create(experiment_name=experiment_name)
    vertex_run = aiplatform.start_run(run=run_name, tensorboard=tensorboard_log_dir)
    
    # Log hyperparameters
    vertex_run.log_params(config.hparams_dict)
    
    # Load data
    X = np.load(preprocessed_npy.path)
    print(f"Loaded data: {X.shape}")
    
    # Create datasets
    train_end, val_end, test_end = calculate_split_indices(X.shape[0])
    train_ds, val_ds, test_ds = create_timeseries_datasets(X, train_end, val_end, test_end)
    
    # Build and compile model
    model = get_model(compile=True)
    
    # Callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=f"{tensorboard_log_dir}/{run_name}",
        histogram_freq=config.HISTOGRAM_FREQ if config.TRACK_HISTOGRAMS else 0,
        write_graph=config.TRACK_GRAPH,
        profile_batch=(config.PROFILE_BATCH_RANGE if config.TRACK_PROFILING else 0)
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1
    )
    
    callbacks = [tensorboard_callback, early_stopping, reduce_lr]
    
    # Train
    print(f"Starting training for {config.EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save(model_output.path)
    print(f"Model saved to: {model_output.path}")
    
    # Save history
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(history_json.path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Log final metrics to Vertex AI
    final_metrics = {
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_route_accuracy': float(history.history['route_output_accuracy'][-1]),
        'final_val_route_accuracy': float(history.history['val_route_output_accuracy'][-1]),
        'final_train_headway_mae': float(history.history['headway_output_mae_seconds'][-1]),
        'final_val_headway_mae': float(history.history['val_headway_output_mae_seconds'][-1]),
    }
    
    vertex_run.log_metrics(final_metrics)
    vertex_run.end_run()
    
    # Log to KFP metrics
    metrics.log_metric('val_loss', final_metrics['final_val_loss'])
    metrics.log_metric('val_route_accuracy', final_metrics['final_val_route_accuracy'])
    metrics.log_metric('val_headway_mae_seconds', final_metrics['final_val_headway_mae'])
    
    print("Training complete!")


# =============================================================================
# Component 4: Evaluation
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def evaluate_component(
    model_input: dsl.Input[dsl.Model],
    preprocessed_npy: dsl.Input[dsl.Dataset],
    metadata_json: dsl.Input[dsl.Artifact],
    evaluation_metrics: dsl.Output[dsl.Metrics],
    evaluation_json: dsl.Output[dsl.Artifact]
):
    """
    Evaluate trained model on test set.
    
    Args:
        model_input: Trained model artifact
        preprocessed_npy: Preprocessed data
        metadata_json: Metadata
        evaluation_metrics: Output metrics
        evaluation_json: Output evaluation results
    """
    import numpy as np
    import json
    import tensorflow as tf
    from sklearn.metrics import f1_score, classification_report
    
    from config import config
    from train import calculate_split_indices, create_timeseries_datasets
    from evaluate import inverse_transform_headway
    
    print("Loading model and data...")
    model = tf.keras.models.load_model(model_input.path)
    X = np.load(preprocessed_npy.path)
    
    # Create test dataset
    train_end, val_end, test_end = calculate_split_indices(X.shape[0])
    _, _, test_ds = create_timeseries_datasets(X, train_end, val_end, test_end)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(test_ds, verbose=1)
    
    # Extract predictions
    y_pred_route = predictions['route_output']
    y_pred_headway = predictions['headway_output']
    
    # Collect ground truth
    y_true_route_list = []
    y_true_headway_list = []
    for _, targets in test_ds:
        y_true_route_list.append(targets['route_output'].numpy())
        y_true_headway_list.append(targets['headway_output'].numpy())
    
    y_true_route = np.concatenate(y_true_route_list, axis=0)
    y_true_headway = np.concatenate(y_true_headway_list, axis=0)
    
    # Trim to match
    min_len = min(len(y_true_route), len(y_pred_route))
    y_true_route = y_true_route[:min_len]
    y_pred_route = y_pred_route[:min_len]
    y_true_headway = y_true_headway[:min_len]
    y_pred_headway = y_pred_headway[:min_len]
    
    # Classification metrics
    y_true_classes = np.argmax(y_true_route, axis=1)
    y_pred_classes = np.argmax(y_pred_route, axis=1)
    f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
    f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    print(f"Route F1 (macro): {f1_macro:.4f}")
    print(f"Route F1 (weighted): {f1_weighted:.4f}")
    
    # Regression metrics
    y_true_minutes = inverse_transform_headway(y_true_headway.flatten())
    y_pred_minutes = inverse_transform_headway(y_pred_headway.flatten())
    
    y_true_seconds = y_true_minutes * 60
    y_pred_seconds = y_pred_minutes * 60
    
    mae_seconds = np.mean(np.abs(y_true_seconds - y_pred_seconds))
    rmse_seconds = np.sqrt(np.mean((y_true_seconds - y_pred_seconds)**2))
    
    print(f"Headway MAE: {mae_seconds:.2f} seconds")
    print(f"Headway RMSE: {rmse_seconds:.2f} seconds")
    
    # Save results
    results = {
        'classification': {
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted)
        },
        'regression': {
            'mae_seconds': float(mae_seconds),
            'rmse_seconds': float(rmse_seconds),
            'mae_minutes': float(mae_seconds / 60),
            'rmse_minutes': float(rmse_seconds / 60)
        }
    }
    
    with open(evaluation_json.path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log to KFP metrics
    evaluation_metrics.log_metric('test_f1_macro', f1_macro)
    evaluation_metrics.log_metric('test_mae_seconds', mae_seconds)
    evaluation_metrics.log_metric('test_rmse_seconds', rmse_seconds)
    
    print("Evaluation complete!")


# =============================================================================
# Pipeline Definition
# =============================================================================

@dsl.pipeline(
    name='a1-headway-prediction-pipeline',
    description='Train and evaluate A1 track headway prediction model',
    pipeline_root=PIPELINE_ROOT
)
def a1_training_pipeline(
    run_name: str = 'run_001',
    project_id: str = config.BQ_PROJECT,
    dataset: str = config.BQ_DATASET,
    table: str = config.BQ_TABLE,
    experiment_name: str = config.EXPERIMENT_NAME,
    tensorboard_log_dir: str = config.TENSORBOARD_LOG_DIR
):
    """
    A1 Training Pipeline.
    
    Args:
        run_name: Unique identifier for this pipeline run
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        experiment_name: Vertex AI experiment name
        tensorboard_log_dir: GCS path for TensorBoard logs
    """
    
    # Step 1: Extract data
    extract_task = extract_data_component(
        project_id=project_id,
        dataset=dataset,
        table=table
    )
    
    # Step 2: Preprocess
    preprocess_task = preprocess_component(
        raw_data_csv=extract_task.outputs['raw_data_csv']
    )
    
    # Step 3: Train
    train_task = train_component(
        preprocessed_npy=preprocess_task.outputs['preprocessed_npy'],
        metadata_json=preprocess_task.outputs['metadata_json'],
        run_name=run_name,
        experiment_name=experiment_name,
        tensorboard_log_dir=tensorboard_log_dir
    )
    
    # Step 4: Evaluate
    evaluate_task = evaluate_component(
        model_input=train_task.outputs['model_output'],
        preprocessed_npy=preprocess_task.outputs['preprocessed_npy'],
        metadata_json=preprocess_task.outputs['metadata_json']
    )


# =============================================================================
# Main Functions
# =============================================================================

def compile_pipeline(output_file: str = 'a1_pipeline.yaml'):
    """Compile pipeline to YAML."""
    print(f"Compiling pipeline to {output_file}...")
    compiler.Compiler().compile(
        pipeline_func=a1_training_pipeline,
        package_path=output_file
    )
    print(f"Pipeline compiled successfully!")


def submit_pipeline(run_name: str = None, enable_caching: bool = False):
    """Submit pipeline to Vertex AI."""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"a1_run_{timestamp}"
    
    print(f"Submitting pipeline run: {run_name}")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Region: {REGION}")
    print(f"  Pipeline root: {PIPELINE_ROOT}")
    
    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}/staging"
    )
    
    # Create pipeline job
    job = aiplatform.PipelineJob(
        display_name=f"a1-training-{run_name}",
        template_path='a1_pipeline.yaml',
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            'run_name': run_name,
            'project_id': PROJECT_ID,
            'dataset': config.BQ_DATASET,
            'table': config.BQ_TABLE,
            'experiment_name': config.EXPERIMENT_NAME,
            'tensorboard_log_dir': config.TENSORBOARD_LOG_DIR
        },
        enable_caching=enable_caching
    )
    
    # Submit
    job.submit()
    
    print(f"\nPipeline submitted!")
    print(f"  Job name: {job.display_name}")
    print(f"  View in console:")
    print(f"  https://console.cloud.google.com/vertex-ai/pipelines/runs/{job.resource_name.split('/')[-1]}?project={PROJECT_ID}")


def main():
    """CLI for pipeline operations."""
    parser = argparse.ArgumentParser(description='A1 Vertex AI Pipeline')
    parser.add_argument('--compile', action='store_true', help='Compile pipeline to YAML')
    parser.add_argument('--submit', action='store_true', help='Submit pipeline to Vertex AI')
    parser.add_argument('--run_name', type=str, help='Unique run identifier')
    parser.add_argument('--enable_caching', action='store_true', help='Enable pipeline caching')
    
    args = parser.parse_args()
    
    if args.compile:
        compile_pipeline()
    
    if args.submit:
        if not args.compile:
            print("Compiling pipeline first...")
            compile_pipeline()
        submit_pipeline(run_name=args.run_name, enable_caching=args.enable_caching)
    
    if not args.compile and not args.submit:
        parser.print_help()


if __name__ == "__main__":
    main()

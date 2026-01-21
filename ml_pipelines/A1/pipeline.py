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
from src.config import config


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
    raw_data_csv: dsl.Output[dsl.Dataset]
):
    """
    Extract A1 track data from BigQuery and save as CSV artifact.
    Uses the extract_data module function.
    
    Args:
        raw_data_csv: Output CSV artifact
    """
    from src.extract_data import extract_a1_data
    
    print(f"Extracting data from BigQuery...")
    
    # Use the existing function from extract_data module
    df = extract_a1_data(output_path=raw_data_csv.path)
    
    print(f"Retrieved {len(df):,} records")
    print(f"Saved to: {raw_data_csv.path}")


# =============================================================================
# Component 2: Preprocessing
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def preprocess_component(
    raw_data_csv: dsl.Input[dsl.Dataset],
    preprocessed_npy: dsl.Output[dsl.Dataset]
):
    """
    Preprocess raw CSV data into model-ready numpy arrays.
    Uses the preprocess_pipeline() function from preprocess module.
    Metadata is saved alongside the .npy file automatically.
    
    Args:
        raw_data_csv: Input CSV artifact
        preprocessed_npy: Output numpy array artifact
    """
    from src.preprocess import preprocess_pipeline
    from pathlib import Path
    
    print(f"Preprocessing data from: {raw_data_csv.path}")
    
    # Kubeflow artifacts are directories - save to file inside
    output_file = Path(preprocessed_npy.path) / 'preprocessed_data.npy'
    
    # Use the existing preprocess_pipeline function
    # It automatically saves metadata as {output_path}_metadata.json
    X, metadata = preprocess_pipeline(
        input_path=raw_data_csv.path,
        output_path=str(output_file)
    )
    
    print(f"Preprocessing complete: {X.shape}")
    print(f"Saved to: {output_file}")
    print(f"Metadata saved alongside data file")


# =============================================================================
# Component 3: Training
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def train_component(
    preprocessed_npy: dsl.Input[dsl.Dataset],
    run_name: str,
    model_output: dsl.Output[dsl.Model],
    history_json: dsl.Output[dsl.Artifact],
    metrics: dsl.Output[dsl.Metrics]
):
    """
    Train stacked GRU model with TensorBoard and Vertex AI Experiments tracking.
    Uses the train module function with callbacks already configured.
    
    Args:
        preprocessed_npy: Input preprocessed data
        run_name: Unique run identifier
        model_output: Output trained model artifact
        history_json: Output training history artifact
        metrics: Output metrics for Vertex AI
    """
    import os
    import numpy as np
    import json
    import shutil
    from pathlib import Path
    
    # Set environment variables for config
    os.environ['GCP_PROJECT_ID'] = 'realtime-headway-prediction'
    os.environ['GCP_REGION'] = 'us-east1'
    os.environ['GCS_BUCKET'] = 'ml-pipelines-headway-prediction'
    os.environ['EXPERIMENT_NAME'] = 'a1-headway-prediction'
    os.environ['BQ_DATASET'] = 'headway_prediction'
    os.environ['BQ_TABLE'] = 'ml'
    
    from src.train import train_model
    from src.config import config
    
    print(f"Training run: {run_name}")
    
    # Copy preprocessed data and metadata to expected location for train_model
    data_dir = Path('data/A1')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Kubeflow artifacts are directories - get file from inside
    source_npy = Path(preprocessed_npy.path) / 'preprocessed_data.npy'
    preprocessed_path = data_dir / 'preprocessed_data.npy'
    shutil.copy(source_npy, preprocessed_path)
    print(f"Copied data from: {source_npy}")
    
    # Copy the metadata file
    metadata_src = str(source_npy).replace('.npy', '_metadata.json')
    metadata_dst = str(preprocessed_path).replace('.npy', '_metadata.json')
    shutil.copy(metadata_src, metadata_dst)
    print(f"Copied metadata from: {metadata_src}")
    
    # Use the existing train_model function with all callbacks configured
    results = train_model(run_name=run_name, use_vertex_experiments=True)
    
    # Copy model to output artifact
    model_path = Path(config.MODEL_ARTIFACTS_DIR) / run_name / 'model'
    shutil.copytree(model_path, model_output.path, dirs_exist_ok=True)
    print(f"Model saved to: {model_output.path}")
    
    # Save history
    with open(history_json.path, 'w') as f:
        json.dump(results['history'], f, indent=2)
    
    # Log to KFP metrics
    history = results['history']
    metrics.log_metric('val_loss', history['val_loss'][-1])
    metrics.log_metric('val_route_accuracy', history['val_route_output_accuracy'][-1])
    metrics.log_metric('val_headway_mae_seconds', history['val_headway_output_mae_seconds'][-1])
    
    print("Training complete!")


# =============================================================================
# Component 4: Evaluation
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def evaluate_component(
    model_input: dsl.Input[dsl.Model],
    preprocessed_npy: dsl.Input[dsl.Dataset],
    run_name: str,
    evaluation_metrics: dsl.Output[dsl.Metrics],
    evaluation_json: dsl.Output[dsl.Artifact]
):
    """
    Evaluate trained model on test set.
    Uses the evaluate_model() function from evaluate module.
    
    Args:
        model_input: Trained model artifact
        preprocessed_npy: Preprocessed data (metadata is alongside)
        run_name: Run identifier for evaluation
        evaluation_metrics: Output metrics
        evaluation_json: Output evaluation results
    """
    import json
    import shutil
    from pathlib import Path
    
    from src.evaluate import evaluate_model
    from src.config import config
    
    print(f"Setting up artifacts for evaluation...")
    
    # Copy model to expected location for evaluate_model
    model_dir = Path(config.MODEL_ARTIFACTS_DIR) / run_name / 'model'
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_input.path, model_dir, dirs_exist_ok=True)
    
    # Copy preprocessed data and metadata to expected location
    data_dir = Path('data/A1')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the .npy file
    preprocessed_path = data_dir / 'preprocessed_data.npy'
    shutil.copy(preprocessed_npy.path, preprocessed_path)
    
    # Copy the metadata file (should be next to the .npy file)
    metadata_src = preprocessed_npy.path.replace('.npy', '_metadata.json')
    metadata_dst = str(preprocessed_path).replace('.npy', '_metadata.json')
    shutil.copy(metadata_src, metadata_dst)
    print(f"Copied metadata from: {metadata_src}")
    
    # Use the existing evaluate_model function
    results = evaluate_model(run_name=run_name)
    
    # Save results to output artifact
    with open(evaluation_json.path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log to KFP metrics
    evaluation_metrics.log_metric('test_f1_macro', results['classification']['f1_macro'])
    evaluation_metrics.log_metric('test_mae_seconds', results['regression']['mae_seconds'])
    evaluation_metrics.log_metric('test_rmse_seconds', results['regression']['rmse_seconds'])
    
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
    run_name: str = 'run_001'
):
    """
    A1 Training Pipeline.
    
    All configuration (project_id, dataset, table, experiment_name, tensorboard_log_dir) 
    is read from config.py by the module functions.
    
    Args:
        run_name: Unique identifier for this pipeline run
    """
    
    # Step 1: Extract data (uses extract_data.py::extract_a1_data)
    extract_task = extract_data_component()
    
    # Step 2: Preprocess (uses preprocess.py::preprocess_pipeline)
    preprocess_task = preprocess_component(
        raw_data_csv=extract_task.outputs['raw_data_csv']
    )
    
    # Step 3: Train (uses train.py::train_model with create_callbacks)
    train_task = train_component(
        preprocessed_npy=preprocess_task.outputs['preprocessed_npy'],
        run_name=run_name
    )
    
    # Configure GPU for training
    train_task.set_accelerator_type('NVIDIA_TESLA_A100')
    train_task.set_accelerator_limit(1)
    train_task.set_cpu_limit('8')
    train_task.set_memory_limit('32G')
    
    # Step 4: Evaluate (uses evaluate.py::evaluate_classification/regression)
    evaluate_task = evaluate_component(
        model_input=train_task.outputs['model_output'],
        preprocessed_npy=preprocess_task.outputs['preprocessed_npy'],
        run_name=run_name
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
    
    # Initialize Vertex AI with experiment tracking
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}/staging",
        experiment=config.EXPERIMENT_NAME
    )
    
    # Create pipeline job
    job = aiplatform.PipelineJob(
        display_name=f"a1-training-{run_name}",
        template_path='a1_pipeline.yaml',
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            'run_name': run_name
        },
        enable_caching=enable_caching
    )
    
    # Submit
    job.submit()
    
    print(f"\nPipeline submitted!")
    print(f"  Job name: {job.display_name}")
    print(f"  View in console:")
    print(f"  https://console.cloud.google.com/vertex-ai/pipelines/runs?project={PROJECT_ID}")


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

"""
Vertex AI Pipeline Example

Example Kubeflow Pipeline (KFP) for orchestrating ML training on Vertex AI.
This serves as a template for creating custom training pipelines.
"""

from typing import NamedTuple
from kfp import dsl
from kfp import compiler


# =============================================================================
# Pipeline Configuration
# =============================================================================

# Update these with your GCP project details
PROJECT_ID = "your-project-id"
REGION = "us-east1"
BUCKET = "your-bucket-name"
PIPELINE_ROOT = f"gs://{BUCKET}/pipeline-runs"
TENSORBOARD_ROOT = f"gs://{BUCKET}/tensorboard"

# Container image with ML dependencies
# Build with: docker build -t <REGION>-docker.pkg.dev/<PROJECT>/ml-pipelines/training:latest .
TRAINING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-pipelines/training:latest"


# =============================================================================
# Pipeline Components
# =============================================================================

@dsl.component(base_image=TRAINING_IMAGE)
def data_extraction_component(
    project_id: str,
    dataset_id: str,
    table_id: str,
    output_data: dsl.Output[dsl.Dataset],
) -> NamedTuple("Outputs", [("num_samples", int), ("num_features", int)]):
    """
    Extract data from BigQuery.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        output_data: Output dataset artifact
        
    Returns:
        Dataset statistics
    """
    from ml_pipelines.data.bigquery_etl import BigQueryETL
    import pandas as pd
    from collections import namedtuple
    
    # Initialize ETL
    etl = BigQueryETL(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id
    )
    
    # Load data
    df = etl.load_data()
    
    # Save to output
    df.to_parquet(output_data.path)
    
    # Return statistics
    Outputs = namedtuple("Outputs", ["num_samples", "num_features"])
    return Outputs(num_samples=len(df), num_features=len(df.columns))


@dsl.component(base_image=TRAINING_IMAGE)
def data_preprocessing_component(
    input_data: dsl.Input[dsl.Dataset],
    train_split: float,
    val_split: float,
    test_split: float,
    train_output: dsl.Output[dsl.Dataset],
    val_output: dsl.Output[dsl.Dataset],
    test_output: dsl.Output[dsl.Dataset],
    scaler_output: dsl.Output[dsl.Artifact],
):
    """
    Preprocess and split data.
    
    Args:
        input_data: Input dataset
        train_split: Training data fraction
        val_split: Validation data fraction
        test_split: Test data fraction
        train_output: Training dataset output
        val_output: Validation dataset output
        test_output: Test dataset output
        scaler_output: Scaler artifact output
    """
    import pandas as pd
    import pickle
    from ml_pipelines.data.bigquery_etl import BigQueryETL
    
    # Load data
    df = pd.read_parquet(input_data.path)
    
    # Initialize ETL
    etl = BigQueryETL(project_id="dummy")
    
    # Split and scale
    train_df, val_df, test_df, scaler = etl.split_and_scale(
        df,
        splits=(train_split, val_split, test_split),
        scaling_method="minmax"
    )
    
    # Save splits
    train_df.to_parquet(train_output.path)
    val_df.to_parquet(val_output.path)
    test_df.to_parquet(test_output.path)
    
    # Save scaler
    with open(scaler_output.path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✓ Data preprocessing complete")


@dsl.component(base_image=TRAINING_IMAGE)
def training_component(
    train_data: dsl.Input[dsl.Dataset],
    val_data: dsl.Input[dsl.Dataset],
    experiment_name: str,
    run_name: str,
    project_id: str,
    tensorboard_log_dir: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    model_output: dsl.Output[dsl.Model],
    metrics_output: dsl.Output[dsl.Metrics],
):
    """
    Train the model with experiment tracking.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        experiment_name: Vertex AI Experiment name
        run_name: Unique run identifier
        project_id: GCP project ID
        tensorboard_log_dir: TensorBoard log directory (gs://)
        batch_size: Training batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        model_output: Trained model output
        metrics_output: Training metrics output
    """
    import pandas as pd
    import tensorflow as tf
    from ml_pipelines.config import ModelConfig, TrackingConfig
    from ml_pipelines.tracking import ExperimentTracker
    from ml_pipelines.training import Trainer
    from ml_pipelines.evaluation.metrics import rmse_seconds, r_squared
    # Import your model architecture here
    # from ml_pipelines.models import YourModel
    
    # Load data
    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)
    
    # TODO: Convert to appropriate format for your model
    # This is a placeholder - adjust based on your model's input requirements
    
    # Create configurations
    model_config = ModelConfig(
        model_name="baseline_model",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    
    tracking_config = TrackingConfig.create_from_model_config(
        model_config=model_config,
        experiment_name=experiment_name,
        run_name=run_name,
        log_dir=tensorboard_log_dir,
        vertex_project=project_id,
    )
    
    # Initialize tracker
    tracker = ExperimentTracker(tracking_config)
    
    # TODO: Build your model
    # model = YourModel(model_config).create()
    
    # Create trainer
    # trainer = Trainer(
    #     model=model,
    #     config=model_config,
    #     tracker=tracker
    # )
    
    # Compile and train
    # trainer.compile(metrics=[rmse_seconds, r_squared])
    # history = trainer.fit(train_dataset, val_dataset)
    
    # Save model
    # trainer.save_model(model_output.path)
    
    # Log final metrics
    # metrics_output.log_metric("final_val_loss", history.history['val_loss'][-1])
    
    tracker.close()
    
    print(f"✓ Training complete")


@dsl.component(base_image=TRAINING_IMAGE)
def evaluation_component(
    test_data: dsl.Input[dsl.Dataset],
    model: dsl.Input[dsl.Model],
    evaluation_output: dsl.Output[dsl.Metrics],
):
    """
    Evaluate model on test set.
    
    Args:
        test_data: Test dataset
        model: Trained model
        evaluation_output: Evaluation metrics output
    """
    import pandas as pd
    import tensorflow as tf
    
    # Load test data
    test_df = pd.read_parquet(test_data.path)
    
    # Load model
    model = tf.keras.models.load_model(model.path)
    
    # TODO: Evaluate model
    # test_results = model.evaluate(test_dataset)
    
    # Log metrics
    # evaluation_output.log_metric("test_loss", test_results[0])
    
    print(f"✓ Evaluation complete")


# =============================================================================
# Pipeline Definition
# =============================================================================

@dsl.pipeline(
    name="ml-training-pipeline",
    description="ML training pipeline with experiment tracking",
    pipeline_root=PIPELINE_ROOT,
)
def ml_training_pipeline(
    project_id: str = PROJECT_ID,
    dataset_id: str = "ml_data",
    table_id: str = "training_data",
    experiment_name: str = "ml-experiment",
    run_name: str = "run-001",
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    batch_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 5e-4,
):
    """
    End-to-end ML training pipeline.
    
    Pipeline steps:
        1. Extract data from BigQuery
        2. Preprocess and split data
        3. Train model with experiment tracking
        4. Evaluate on test set
    """
    # Step 1: Data extraction
    data_extraction_task = data_extraction_component(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
    )
    
    # Step 2: Data preprocessing
    preprocessing_task = data_preprocessing_component(
        input_data=data_extraction_task.outputs["output_data"],
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )
    
    # Step 3: Model training
    training_task = training_component(
        train_data=preprocessing_task.outputs["train_output"],
        val_data=preprocessing_task.outputs["val_output"],
        experiment_name=experiment_name,
        run_name=run_name,
        project_id=project_id,
        tensorboard_log_dir=f"{TENSORBOARD_ROOT}/{experiment_name}/{run_name}",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    
    # Step 4: Model evaluation
    evaluation_task = evaluation_component(
        test_data=preprocessing_task.outputs["test_output"],
        model=training_task.outputs["model_output"],
    )


# =============================================================================
# Pipeline Compilation and Submission
# =============================================================================

def compile_pipeline(output_path: str = "ml_training_pipeline.json"):
    """
    Compile the pipeline to JSON.
    
    Args:
        output_path: Path to save compiled pipeline
    """
    compiler.Compiler().compile(
        pipeline_func=ml_training_pipeline,
        package_path=output_path
    )
    print(f"✓ Pipeline compiled to: {output_path}")


def submit_pipeline(
    pipeline_json_path: str = "ml_training_pipeline.json",
    experiment_name: str = "ml-experiment",
    run_name: str = "run-001",
):
    """
    Submit pipeline to Vertex AI.
    
    Args:
        pipeline_json_path: Path to compiled pipeline JSON
        experiment_name: Experiment name
        run_name: Run name
    """
    from google.cloud import aiplatform
    
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
    )
    
    job = aiplatform.PipelineJob(
        display_name=f"{experiment_name}-{run_name}",
        template_path=pipeline_json_path,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "experiment_name": experiment_name,
            "run_name": run_name,
        },
    )
    
    job.submit()
    print(f"✓ Pipeline submitted: {job.resource_name}")
    print(f"  Monitor at: {job._dashboard_uri()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument("--compile", action="store_true", help="Compile pipeline")
    parser.add_argument("--submit", action="store_true", help="Submit pipeline to Vertex AI")
    parser.add_argument("--experiment-name", default="ml-experiment", help="Experiment name")
    parser.add_argument("--run-name", default="run-001", help="Run name")
    
    args = parser.parse_args()
    
    if args.compile:
        compile_pipeline()
    
    if args.submit:
        submit_pipeline(
            experiment_name=args.experiment_name,
            run_name=args.run_name
        )

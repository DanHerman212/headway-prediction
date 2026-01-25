
import os
from kfp import dsl
from kfp import compiler
from dotenv import dotenv_values

# Load environment variables from .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
config = dotenv_values(env_path)

# Extract Image URI from config
# In a real setup, this should be the image built from the Dockerfile
IMAGE_URI = config.get("TENSORFLOW_IMAGE_URI", "us-docker.pkg.dev/headway-prediction/ml-pipelines/headway-training:latest")
PIPELINE_ROOT = config.get("PIPELINE_ROOT", "gs://headway-prediction-pipelines/root")

@dsl.container_component
def extract_op(
    output_data: dsl.OutputPath("CSV")
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/extract.py"],
        args=[
            "--output_path", output_data
        ]
    )

@dsl.container_component
def preprocess_op(
    input_data: dsl.InputPath("CSV"),
    output_data: dsl.OutputPath("CSV")
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/preprocess.py"],
        args=[
            "--input_path", input_data,
            "--output_path", output_data
        ]
    )

@dsl.container_component
def train_op(
    input_data: dsl.InputPath("CSV"),
    model_output: dsl.OutputPath("Model"),
    test_data_output: dsl.OutputPath("CSV"),
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/train.py"],
        args=[
            "--input_path", input_data,
            "--model_output_path", model_output,
            "--test_data_output_path", test_data_output
        ]
    )

@dsl.container_component
def eval_op(
    model_input: dsl.InputPath("Model"),
    test_data_input: dsl.InputPath("CSV"),
    metrics_output: dsl.OutputPath("JSON")
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/eval.py"],
        args=[
            "--model_path", model_input,
            "--test_data_path", test_data_input,
            "--output_metrics_path", metrics_output
        ]
    )

@dsl.pipeline(
    name="headway-prediction-pipeline-v2",
    description="End-to-end headway prediction with clean architecture",
    pipeline_root=PIPELINE_ROOT,
)
def headway_pipeline(
    project_id: str = config.get("GCP_PROJECT_ID", ""),
    region: str = config.get("VERTEX_LOCATION", "us-east1"),
):
    # 1. Extract
    extract_task = extract_op()
    
    # Apply env vars to task
    for key, val in config.items():
        extract_task.set_env_variable(key, val)
        
    # 2. Preprocess
    preprocess_task = preprocess_op(
        input_data=extract_task.outputs['output_data']
    )
    for key, val in config.items():
        preprocess_task.set_env_variable(key, val)
        
    # 3. Train
    train_task = train_op(
        input_data=preprocess_task.outputs['output_data']
    )
    # Configure A100 GPU (requires A2 machine series)
    train_task.set_machine_type("a2-highgpu-1g")
    train_task.set_gpu_limit(1) 
    
    for key, val in config.items():
        train_task.set_env_variable(key, val)
        
    # 4. Evaluate
    eval_task = eval_op(
        model_input=train_task.outputs['model_output'],
        test_data_input=train_task.outputs['test_data_output']
    )
    for key, val in config.items():
        eval_task.set_env_variable(key, val)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=headway_pipeline,
        package_path="headway_pipeline.json"
    )
    print("Pipeline compiled to headway_pipeline.json")

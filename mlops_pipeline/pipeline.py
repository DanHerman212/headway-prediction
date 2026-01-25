import os
from kfp import dsl
from kfp import compiler
from dotenv import dotenv_values

# Load environment variables from .env
# We use a safe loading mechanism to prevent crashes if file is missing
env_path = os.path.join(os.path.dirname(__file__), '.env')
file_config = dotenv_values(env_path) if os.path.exists(env_path) else {}

# Helper to get config from Env Vars first, then .env file, then default
def get_config(key, default):
    return os.environ.get(key, file_config.get(key, default))

# Defaults
IMAGE_URI = get_config("TENSORFLOW_IMAGE_URI", "us-docker.pkg.dev/headway-prediction/ml-pipelines/headway-training:latest")
PIPELINE_ROOT = get_config("PIPELINE_ROOT", "gs://headway-prediction-pipelines/root")
PROJECT_ID = get_config("GCP_PROJECT_ID", "")
REGION = get_config("VERTEX_LOCATION", "us-east1")

# Use file_config for iterating over env vars, but we could also merge os.environ if needed.
# For now, sticking to file_config for iteration is safer to avoid polluting with system env vars.
config = file_config

@dsl.container_component
def extract_op(output_data: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/extract.py"],
        args=[
            "--output_path", output_data.path
        ]
    )

@dsl.container_component
def preprocess_op(input_data: dsl.Input[dsl.Dataset], output_data: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/preprocess.py"],
        args=[
            "--input_path", input_data.path,
            "--output_path", output_data.path
        ]
    )

@dsl.container_component
def train_op(
    input_data: dsl.Input[dsl.Dataset],
    model_output: dsl.Output[dsl.Model],
    test_data_output: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/train.py"],
        args=[
            "--input_path", input_data.path,
            "--model_output_path", model_output.path,
            "--test_data_output_path", test_data_output.path
        ]
    )

@dsl.container_component
def eval_op(
    model_input: dsl.Input[dsl.Model],
    test_data_input: dsl.Input[dsl.Dataset],
    metrics_output: dsl.Output[dsl.Metrics]
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/eval.py"],
        args=[
            "--model_path", model_input.path,
            "--test_data_path", test_data_input.path,
            "--output_metrics_path", metrics_output.path
        ]
    )

@dsl.pipeline(
    name="headway-prediction-pipeline-v2",
    description="End-to-end headway prediction with clean architecture",
    pipeline_root=PIPELINE_ROOT,
)
def headway_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION,
):
    # 1. Extract
    extract_task = extract_op()
    extract_task.set_caching_options(False) # Force re-execution to pick up code changes
    extract_task.set_env_variable("PYTHONPATH", "/app") # Ensure src module is found
    for key, val in config.items():
        if val: # SAFETY CHECK: Prevent crash if .env has empty keys
            extract_task.set_env_variable(key, val)
        
    # 2. Preprocess
    preprocess_task = preprocess_op(
        input_data=extract_task.outputs['output_data']
    )
    preprocess_task.set_caching_options(False)
    preprocess_task.set_env_variable("PYTHONPATH", "/app")
    for key, val in config.items():
        if val:
            preprocess_task.set_env_variable(key, val)
        
    # 3. Train
    train_task = train_op(
        input_data=preprocess_task.outputs['output_data']
    )
    train_task.set_caching_options(False)
    
    # Configure A100 GPU
    train_task.set_gpu_limit(1)
    train_task.set_accelerator_type("NVIDIA_TESLA_A100")
    train_task.set_env_variable("PYTHONPATH", "/app")
    
    for key, val in config.items():
        if val:
            train_task.set_env_variable(key, val)
        
    # 4. Evaluate
    eval_task = eval_op(
        model_input=train_task.outputs['model_output'],
        test_data_input=train_task.outputs['test_data_output']
    )
    eval_task.set_env_variable("PYTHONPATH", "/app")
    for key, val in config.items():
        if val:
            eval_task.set_env_variable(key, val)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=headway_pipeline,
        package_path="headway_pipeline.json"
    )
    print("Pipeline compiled to headway_pipeline.json")

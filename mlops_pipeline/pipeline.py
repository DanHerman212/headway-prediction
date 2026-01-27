import os
import yaml
from kfp import dsl
from kfp import compiler
from dotenv import dotenv_values

# Load environment variables from .env
# We use a safe loading mechanism to prevent crashes if file is missing
env_path = os.path.join(os.path.dirname(__file__), '.env')
file_config = dotenv_values(env_path) if os.path.exists(env_path) else {}

# Load run_name from config.yaml
config_path = os.path.join(os.path.dirname(__file__), 'conf/config.yaml')
DEFAULT_RUN_NAME = "manual-run"
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            DEFAULT_RUN_NAME = yaml_config.get('experiment', {}).get('run_name', "manual-run")
    except Exception as e:
        print(f"Warning: Failed to load config.yaml: {e}")

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
def extract_op(output_data: dsl.Output[dsl.Dataset], run_name: str):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/extract.py"],
        args=[
            f"paths.output_path={output_data.path}",
            f"experiment.run_name={run_name}"
        ]
    )

@dsl.container_component
def preprocess_op(input_data: dsl.Input[dsl.Dataset], output_data: dsl.Output[dsl.Dataset], run_name: str):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/preprocess.py"],
        args=[
            f"paths.input_path={input_data.path}",
            f"paths.output_path={output_data.path}",
            f"experiment.run_name={run_name}"
        ]
    )

@dsl.container_component
def train_op(
    input_data: dsl.Input[dsl.Dataset],
    model_output: dsl.Output[dsl.Model],
    test_data_output: dsl.Output[dsl.Dataset],
    run_name: str
):
    # Using Hydra syntax for overrides: key=value
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/train.py"],
        args=[
            f"paths.input_path={input_data.path}",
            f"paths.model_output_path={model_output.path}",
            f"paths.test_data_output_path={test_data_output.path}",
            f"experiment.run_name={run_name}"
        ]
    )

@dsl.container_component
def eval_op(
    model_input: dsl.Input[dsl.Model],
    test_data_input: dsl.Input[dsl.Dataset],
    metrics_output: dsl.Output[dsl.Metrics],
    html_report: dsl.Output[dsl.HTML],
    run_name: str
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/eval.py"],
        args=[
            f"paths.model_path={model_input.path}",
            f"paths.test_data_path={test_data_input.path}",
            f"paths.output_metrics_path={metrics_output.path}",
            f"paths.output_html_path={html_report.path}",
            f"experiment.run_name={run_name}"
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
    run_name: str = DEFAULT_RUN_NAME
):
    # 1. Extract
    extract_task = extract_op(run_name=run_name)
    extract_task.set_caching_options(False) # Force re-execution to pick up code changes
    extract_task.set_env_variable("PYTHONPATH", "/app") # Ensure src module is found
    
    for key, val in config.items():
        if val: # SAFETY CHECK: Prevent crash if .env has empty keys
            extract_task.set_env_variable(key, str(val))
        
    # 2. Preprocess
    preprocess_task = preprocess_op(
        input_data=extract_task.outputs['output_data'],
        run_name=run_name
    )
    preprocess_task.set_caching_options(False)
    preprocess_task.set_env_variable("PYTHONPATH", "/app")
    for key, val in config.items():
        if val:
            preprocess_task.set_env_variable(key, str(val))
        
    # 3. Train
    train_task = train_op(
        input_data=preprocess_task.outputs['output_data'],
        run_name=run_name
    )
    train_task.set_caching_options(False)
    
    # Configure A100 GPU
    train_task.set_gpu_limit(1)
    train_task.set_accelerator_type("NVIDIA_TESLA_A100")
    train_task.set_env_variable("PYTHONPATH", "/app")
    
    for key, val in config.items():
        if val:
            train_task.set_env_variable(key, str(val))
        
    # 4. Evaluate
    eval_task = eval_op(
        model_input=train_task.outputs['model_output'],
        test_data_input=train_task.outputs['test_data_output'],
        run_name=run_name
    )
    eval_task.set_env_variable("PYTHONPATH", "/app")
    for key, val in config.items():
        if val:
            eval_task.set_env_variable(key, str(val))

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=headway_pipeline,
        package_path="headway_pipeline.json"
    )
    print("Pipeline compiled to headway_pipeline.json")

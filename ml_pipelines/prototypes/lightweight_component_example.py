from kfp import dsl
from kfp import compiler

# ==============================================================================
# LIGHTWEIGHT PYTHON COMPONENT
# ==============================================================================
# Advantages:
# 1. No need to build a Docker image for every code change.
# 2. Function source code is pickled/serialized directly into the pipeline YAML.
# 3. Can define dependencies via `packages_to_install` list.
# 4. Great for "Glue Code", simple data transformation, or metrics calculation.
# ==============================================================================

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "google-cloud-storage"]
)
def train_lightweight_model(
    project_id: str,
    epochs: int,
    metrics_output: dsl.Output[dsl.Metrics]
) -> str:
    # --------------------------------------------------------------------------
    # ALL code must be inside the function. No outside imports allowed.
    # --------------------------------------------------------------------------
    import json
    import logging
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    logging.info(f"Starting training for project: {project_id}")
    
    # 1. Simulate Data Loading (Fast Iteration)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    # 2. Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Log Metrics
    score = model.score(X, y)
    metrics_output.log_metric("r2_score", score)
    
    logging.info(f"Training complete. Score: {score}")
    
    # Return artifacts
    return f"Model trained with {epochs} epochs (simulated)"

# ==============================================================================
# PIPELINE DEFINITION
# ==============================================================================
@dsl.pipeline(
    name="lightweight-prototype-pipeline",
    description="Demonstrates fast iteration without docker builds"
)
def prototype_pipeline(project_id: str):
    
    # Call the component directly as a function
    train_op = train_lightweight_model(
        project_id=project_id,
        epochs=10
    )
    
    # You can still set resources
    train_op.set_cpu_limit("1")
    train_op.set_memory_limit("500M")

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=prototype_pipeline,
        package_path="lightweight_prototype.json"
    )
    print("Compiled to lightweight_prototype.json")

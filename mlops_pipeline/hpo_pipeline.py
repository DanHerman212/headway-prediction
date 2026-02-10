"""
hpo_pipeline.py
---------------
Standalone ZenML pipeline for Hyperparameter Optimization.

Flow:
1. Load Config (HPO profile)
2. Ingest Data
3. Process Data
4. Run Vizier Study (returns best params)

It uses the same data processing steps as the main pipeline, ensuring consistency.
"""

from typing import List, Optional

from zenml import pipeline
from zenml.config import DockerSettings

# Import existing steps
from .src.steps.config_loader import load_config_step
from .src.steps.ingest_data import ingest_data_step
from .src.steps.process_data import process_data_step

# Import the new HPO step
from .src.steps.run_vizier_study import run_vizier_study_step

# Reuse the same Docker settings (CUDA, etc.)
docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
    requirements="mlops_pipeline/requirements.txt",
    replicate_local_python_environment=False,
    python_package_installer="pip",
)

@pipeline(
    enable_cache=False,
    settings={"docker": docker_settings}
)
def headway_hpo_pipeline(
    data_path: str,
    hydra_overrides: Optional[List[str]] = None,
):
    """
    Pipeline that runs a Vizier HPO study.
    """
    
    # Force the HPO training profile (fast trials)
    # This override is applied on top of whatever user passed
    overrides = ["training=hpo"]
    
    # Also inject the search space into the main config tree dynamically
    # This removes the need for a second load_config_step
    overrides.append("+hpo_search_space=vizier_v1")

    if hydra_overrides:
        overrides.extend(hydra_overrides)

    # 1. Load Config (Execution + Search Space Combined)
    config = load_config_step(overrides=overrides)

    # 2. Ingest Data
    raw_df = ingest_data_step(file_path=data_path)

    # 3. Process Data
    train_ds, val_ds, _ = process_data_step(
        raw_data=raw_df, 
        config=config
    )

    # 4. Run Vizier
    best_params = run_vizier_study_step(
        training_dataset=train_ds,
        validation_dataset=val_ds,
        config=config
    )
    
    return best_params

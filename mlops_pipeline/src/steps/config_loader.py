import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from zenml import step
import os

@step
def load_config_step(config_name: str = "config", config_path: str = "../../conf") -> DictConfig:
    """
    Loads the Hydra configuration for the pipeline.
    """
    # 1. Clear any existing Hydra instance (crucial for recurring pipeline runs)
    GlobalHydra.instance().clear()

    # 2. Resolve absolute path to config if needed, or use relative
    # ZenML runs might change CWD, so let's be safe if we can.
    # For now, we trust the relative path from the source root.
    
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name=config_name)
    
    return cfg
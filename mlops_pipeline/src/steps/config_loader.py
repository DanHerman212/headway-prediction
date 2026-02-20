import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from typing import List, Optional


def load_config(
    config_name: str = "config",
    config_path: str = "../../conf",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Loads the Hydra configuration.
    YAML files in conf/ provide version-controlled defaults.
    Use the 'overrides' parameter for experiment-specific changes
    without triggering a Docker rebuild.
    """
    GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])

    return cfg
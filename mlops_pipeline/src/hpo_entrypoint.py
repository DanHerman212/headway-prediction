"""
hpo_entrypoint.py
-----------------
The execution script for a single HPO trial.
Runs inside a Vertex AI custom container.

Responsibilities:
1. Accept generic command-line arguments (arbitrary key=value pairs).
2. Load the cached TimeSeriesDataSets from GCS.
3. Compose the Hydra configuration, applying the CLI arguments as overrides.
4. Execute the training using `src.training_core`.
5. Report the result (val_loss) to Vizier via `cloudml-hypertune`.
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

import hydra
import hypertune
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_forecasting import TimeSeriesDataSet

# Add project root to path so we can import src.training_core
# We assume this script runs as a module: python -m mlops_pipeline.src.hpo_entrypoint
# But if run directly, we might need this path hack.
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from mlops_pipeline.src.training_core import train_tft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parses known args (dataset paths) and treats the rest as Hydra overrides.
    Example:
      --train_path gs://... --lr=0.01 --batch_size=64
    Becomes:
      args.train_path, overrides=["lr=0.01", "batch_size=64"]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)
    
    known_args, unknown_args = parser.parse_known_args()
    
    # Convert unknown args (like --learning_rate 0.01) into Hydra overrides (learning_rate=0.01)
    # Vizier passes args as flags: --param_name value
    hydra_overrides = []
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]  # remove --
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                value = unknown_args[i + 1]
                i += 1
            else:
                value = "true"  # flag assumption
            
            # Map bare param names to Hydra keys if necessary
            # For simplicity, we assume the Vizier parameter names match Hydra keys exactly
            # e.g. "model.learning_rate" -> model.learning_rate=0.01
            hydra_overrides.append(f"{key}={value}")
        i += 1
        
    return known_args, hydra_overrides


def load_dataset(gcs_path: str) -> TimeSeriesDataSet:
    """Loads a serialized TimeSeriesDataSet from GCS."""
    import gcsfs  # Implicit dependency for torch.load on gs:// paths
    logger.info(f"Loading dataset from {gcs_path}...")
    
    # We use GCSFileSystem directly to be robust
    fs = gcsfs.GCSFileSystem()
    with fs.open(gcs_path, "rb") as f:
        ds = torch.load(f, weights_only=False)
    
    logger.info(f"Loaded dataset. Length: {len(ds)}")
    return ds


def main():
    args, overrides = parse_args()
    
    logger.info(f"Received overrides from Vizier: {overrides}")

    # 1. Load Data
    train_ds = load_dataset(args.train_dataset_path)
    val_ds = load_dataset(args.val_dataset_path)

    # 2. Build Config
    # We enforce 'training=hpo' to ensure we use the fast trial profile
    base_overrides = ["training=hpo"] + overrides
    
    # Initialize Hydra relative to this file's location
    # Conf is at ../../conf relative to mlops_pipeline/src/hpo_entrypoint.py
    config_path = "../../conf" 
    
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name="config", overrides=base_overrides)

    logger.info(f"Effective Config:\n{OmegaConf.to_yaml(config)}")

    # 3. Train
    # We pass None for logger -> defaults to CSVLogger (safe for container)
    result = train_tft(
        training_dataset=train_ds,
        validation_dataset=val_ds,
        config=config,
        lightning_logger=None, 
    )

    # 4. Report to Vizier
    val_loss = result.best_val_loss
    logger.info(f"Reporting metric val_loss={val_loss} to Vizier...")
    
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="val_loss",
        metric_value=val_loss,
        global_step=config.training.max_epochs
    )
    
    logger.info("Trial complete.")


if __name__ == "__main__":
    main()

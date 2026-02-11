"""
hpo_entrypoint.py
-----------------
Execution script for a single Vertex AI HPO trial.

This is the ENTRYPOINT of the Docker container.  Vizier launches it
once per trial, appending hyperparameter flags (``--key=value``) to
the command line.

Flow:
    1. Parse CLI args  →  dataset paths + Hydra overrides
    2. Load cached TimeSeriesDataSets from GCS
    3. Compose Hydra config with the overrides
    4. Train via training_core.train_tft()
    5. Report val_loss to Vizier via cloudml-hypertune
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing  (stdlib only — no heavy deps, safe to import in tests)
# ---------------------------------------------------------------------------

def vizier_args_to_hydra_overrides(unknown_args: List[str]) -> List[str]:
    """
    Convert Vizier CLI flags into Hydra override strings.

    Vizier passes hyperparameters as ``--key=value``.  Stripping the
    ``--`` prefix yields a valid Hydra override directly.
    """
    overrides: List[str] = []
    for token in unknown_args:
        if token.startswith("--") and "=" in token:
            overrides.append(token[2:])          # --key=value → key=value
        elif token.startswith("--"):
            overrides.append(f"{token[2:]}=true") # bare flag fallback
    return overrides


def parse_and_convert_args(
    args_list: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    """Parse known args (dataset paths) and convert the rest to Hydra overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)

    known, unknown = parser.parse_known_args(args_list)
    return known, vizier_args_to_hydra_overrides(unknown)


# ---------------------------------------------------------------------------
# Main  (heavy imports are scoped here so the module stays lightweight)
# ---------------------------------------------------------------------------

def main() -> None:
    import hydra
    import hypertune
    import torch
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    # Ensure mlops_pipeline is importable
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, proj_root)

    from mlops_pipeline.src.training_core import train_tft

    # Conf dir: mlops_pipeline/conf (sibling of src/)
    conf_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "conf"))

    # 1. Parse CLI
    args, hydra_overrides = parse_and_convert_args()
    logger.info("Dataset paths — train: %s  val: %s",
                args.train_dataset_path, args.val_dataset_path)
    logger.info("Hydra overrides from Vizier: %s", hydra_overrides)

    # 2. Load data
    import gcsfs
    fs = gcsfs.GCSFileSystem()

    logger.info("Loading training dataset …")
    with fs.open(args.train_dataset_path, "rb") as f:
        train_ds = torch.load(f, weights_only=False)

    logger.info("Loading validation dataset …")
    with fs.open(args.val_dataset_path, "rb") as f:
        val_ds = torch.load(f, weights_only=False)

    # 3. Compose config
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=conf_dir, version_base=None):
        config = hydra.compose(config_name="config",
                               overrides=["training=hpo"] + hydra_overrides)
    logger.info("Effective config:\n%s", OmegaConf.to_yaml(config))

    # 4. Train — use TensorBoardLogger so pytorch-forecasting's
    #    add_embedding/add_histogram calls don't crash.
    #    Each trial gets its own subdirectory via CLOUD_ML_TRIAL_ID so
    #    TensorBoard's HParams dashboard can compare them side by side.
    from lightning.pytorch.loggers import TensorBoardLogger
    trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", "local")
    tb_logger = TensorBoardLogger(
        save_dir=config.training.tensorboard_log_dir,
        name="hpo_trials",
        version=f"trial_{trial_id}",
        default_hp_metric=False,
    )

    # Log hyperparams so TensorBoard HParams plugin can display them
    hparams = {
        "learning_rate": config.model.learning_rate,
        "hidden_size": config.model.hidden_size,
        "attention_head_size": config.model.attention_head_size,
        "dropout": config.model.dropout,
        "hidden_continuous_size": config.model.hidden_continuous_size,
        "batch_size": config.training.batch_size,
    }
    tb_logger.log_hyperparams(hparams)

    result = train_tft(
        training_dataset=train_ds,
        validation_dataset=val_ds,
        config=config,
        lightning_logger=tb_logger,
    )

    # 5. Log final metric to TensorBoard HParams + report to Vizier
    val_loss = result.best_val_loss
    tb_logger.log_metrics({"hp/val_loss": val_loss})
    tb_logger.finalize("success")

    logger.info("Reporting val_loss=%f to Vizier", val_loss)
    reporter = hypertune.HyperTune()
    reporter.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="val_loss",
        metric_value=val_loss,
        global_step=config.training.max_epochs,
    )
    logger.info("Trial complete.")


if __name__ == "__main__":
    main()

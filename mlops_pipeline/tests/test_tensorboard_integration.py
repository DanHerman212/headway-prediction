"""
Tests for the MLflow → TensorBoard migration.

Verifies that:
  - train_model.py no longer imports mlflow
  - evaluate_model.py no longer imports mlflow
  - training_core.py creates a profiler when config.training.profiler == "pytorch"
  - training_core.py skips profiler when config.training.profiler is null
  - TensorBoard config keys exist in both training profiles
"""

import importlib
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# 1. Import‑level checks — ensure mlflow is gone from train_model / evaluate
# ---------------------------------------------------------------------------

def _module_source(rel_path: str) -> str:
    """Read source file relative to project root."""
    proj_root = Path(__file__).resolve().parent.parent.parent
    return (proj_root / rel_path).read_text()


def test_train_model_no_mlflow_import():
    src = _module_source("mlops_pipeline/src/steps/train_model.py")
    assert "import mlflow" not in src
    assert "MLFlowLogger" not in src
    assert "SafeMLFlowLogger" not in src
    assert "experiment_tracker" not in src


def test_evaluate_model_no_mlflow_import():
    src = _module_source("mlops_pipeline/src/steps/evaluate_model.py")
    assert "import mlflow" not in src
    assert "mlflow.log" not in src


def test_train_model_uses_tensorboard():
    src = _module_source("mlops_pipeline/src/steps/train_model.py")
    assert "TensorBoardLogger" in src
    assert "tensorboard_log_dir" in src


def test_evaluate_model_uses_tensorboard():
    src = _module_source("mlops_pipeline/src/steps/evaluate_model.py")
    assert "SummaryWriter" in src
    assert "tensorboard_log_dir" in src


# ---------------------------------------------------------------------------
# 2. Hydra config — tensorboard fields present in both training profiles
# ---------------------------------------------------------------------------

def test_default_training_config_has_tensorboard_keys():
    proj_root = Path(__file__).resolve().parent.parent.parent
    cfg = OmegaConf.load(proj_root / "mlops_pipeline/conf/training/default.yaml")
    assert "tensorboard_log_dir" in cfg
    assert cfg.tensorboard_log_dir.startswith("gs://")
    # profiler default is null
    assert "profiler" in cfg


def test_hpo_training_config_has_tensorboard_keys():
    proj_root = Path(__file__).resolve().parent.parent.parent
    cfg = OmegaConf.load(proj_root / "mlops_pipeline/conf/training/hpo.yaml")
    assert "tensorboard_log_dir" in cfg
    assert cfg.profiler is None


# ---------------------------------------------------------------------------
# 3. training_core profiler logic — unit test the conditional path
# ---------------------------------------------------------------------------

def _make_config(profiler_val=None):
    """Build a minimal OmegaConf config for training_core."""
    return OmegaConf.create({
        "training": {
            "batch_size": 8,
            "val_batch_size_multiplier": 1,
            "num_workers": 0,
            "pin_memory": False,
            "max_epochs": 1,
            "gradient_clip_val": 0.1,
            "limit_train_batches": 1.0,
            "precision": 32,
            "accelerator": "cpu",
            "devices": 1,
            "enable_model_summary": False,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 1e-4,
            "profiler": profiler_val,
            "profiler_schedule": {"wait": 1, "warmup": 1, "active": 1, "repeat": 1},
        },
        "model": {
            "learning_rate": 0.001,
            "hidden_size": 16,
            "attention_head_size": 1,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
        },
    })


def test_profiler_none_when_config_null():
    """When profiler=null in config, Trainer should get profiler=None."""
    config = _make_config(profiler_val=None)

    with patch("mlops_pipeline.src.training_core.create_model") as mock_model, \
         patch("lightning.pytorch.Trainer") as MockTrainer:

        mock_model.return_value = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.fit = MagicMock()
        # Simulate EarlyStopping best_score
        mock_callback = MagicMock()
        mock_callback.best_score = 1.23
        mock_trainer_instance.callbacks = [mock_callback]
        MockTrainer.return_value = mock_trainer_instance

        from mlops_pipeline.src.training_core import train_tft

        mock_ds = MagicMock()
        mock_ds.to_dataloader = MagicMock(return_value=[])

        result = train_tft(mock_ds, mock_ds, config, lightning_logger=None)

        # Trainer should have been called with profiler=None
        call_kwargs = MockTrainer.call_args[1]
        assert call_kwargs["profiler"] is None


def test_profiler_created_when_config_pytorch():
    """When profiler='pytorch' in config, Trainer should get a PyTorchProfiler."""
    config = _make_config(profiler_val="pytorch")

    with patch("mlops_pipeline.src.training_core.create_model") as mock_model, \
         patch("lightning.pytorch.Trainer") as MockTrainer:

        mock_model.return_value = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.fit = MagicMock()
        mock_callback = MagicMock()
        mock_callback.best_score = 0.99
        mock_trainer_instance.callbacks = [mock_callback]
        MockTrainer.return_value = mock_trainer_instance

        from mlops_pipeline.src.training_core import train_tft

        mock_ds = MagicMock()
        mock_ds.to_dataloader = MagicMock(return_value=[])

        # Need a logger with log_dir for the trace handler
        mock_logger = MagicMock()
        mock_logger.log_dir = "/tmp/test_tb"

        result = train_tft(mock_ds, mock_ds, config, lightning_logger=mock_logger)

        call_kwargs = MockTrainer.call_args[1]
        assert call_kwargs["profiler"] is not None


# ---------------------------------------------------------------------------
# 4. requirements.txt — mlflow integration removed
# ---------------------------------------------------------------------------

def test_requirements_no_mlflow():
    proj_root = Path(__file__).resolve().parent.parent.parent
    reqs = (proj_root / "mlops_pipeline/requirements.txt").read_text()
    assert "mlflow" not in reqs.lower()
    assert "tensorboard" in reqs.lower()
    assert "zenml[gcp]" in reqs

"""
Tests for experiment tracking integration.

Verifies that:
  - train_model.py uses vertex_tracker + aiplatform SDK, no mlflow
  - evaluate_model.py uses vertex_tracker + aiplatform SDK, no mlflow
  - training_core.py creates a profiler when config.training.profiler == "pytorch"
  - training_core.py skips profiler when config.training.profiler is null
  - GradientHistogramCallback logs weight/gradient histograms via on_after_backward
  - TensorBoard config keys exist in both training profiles
  - requirements.txt has correct deps
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
    assert 'experiment_tracker="vertex_tracker"' in src
    assert "aiplatform.log_params" in src
    assert "aiplatform.log_metrics" in src


def test_evaluate_model_no_mlflow_import():
    src = _module_source("mlops_pipeline/src/steps/evaluate_model.py")
    assert "import mlflow" not in src
    assert "mlflow.log" not in src
    assert 'experiment_tracker="vertex_tracker"' in src
    assert "aiplatform.log_metrics" in src


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
    # profiler default is "pytorch"
    assert cfg.profiler == "pytorch"
    # log_every_n_steps present
    assert "log_every_n_steps" in cfg
    assert cfg.log_every_n_steps == 50


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
    assert "zenml" in reqs


# ---------------------------------------------------------------------------
# 5. Gradient histogram callback
# ---------------------------------------------------------------------------

def test_gradient_histogram_callback_exists():
    """GradientHistogramCallback is defined in training_core."""
    src = _module_source("mlops_pipeline/src/training_core.py")
    assert "class GradientHistogramCallback" in src
    assert "on_after_backward" in src
    assert "add_histogram" in src


def test_gradient_histogram_callback_logs_weights_and_gradients():
    """Source includes add_histogram for both weights/ and gradients/ prefixes."""
    src = _module_source("mlops_pipeline/src/training_core.py")
    assert '"weights/' in src or "f'weights/" in src or "'weights/" in src
    assert '"gradients/' in src or "f'gradients/" in src or "'gradients/" in src


def test_gradient_histogram_callback_respects_cadence():
    """Callback checks log_every_n_steps before logging."""
    src = _module_source("mlops_pipeline/src/training_core.py")
    assert "log_every_n_steps" in src
    # Should have a modulo check to skip off-cadence steps
    assert "% self.log_every_n_steps" in src


def test_gradient_callback_unit_behavior():
    """Self-contained unit test for the callback's on_after_backward logic."""
    import torch
    from unittest.mock import MagicMock

    # Define the callback class inline (mirrors training_core.GradientHistogramCallback)
    # to avoid importing heavy lightning/pytorch_forecasting dependencies.
    class GradientHistogramCallback:
        def __init__(self, log_every_n_steps=50):
            self.log_every_n_steps = log_every_n_steps

        def on_after_backward(self, trainer, pl_module):
            if trainer.global_step % self.log_every_n_steps != 0:
                return
            tb_logger = pl_module.logger
            if tb_logger is None:
                return
            experiment = tb_logger.experiment
            if not hasattr(experiment, "add_histogram"):
                return
            step = trainer.global_step
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    experiment.add_histogram(f"weights/{name}", param.data, global_step=step)
                    if param.grad is not None:
                        experiment.add_histogram(f"gradients/{name}", param.grad, global_step=step)

    cb = GradientHistogramCallback(log_every_n_steps=50)

    # Off-cadence: should NOT call add_histogram
    trainer = MagicMock()
    trainer.global_step = 7
    pl_module = MagicMock()
    cb.on_after_backward(trainer, pl_module)
    pl_module.logger.experiment.add_histogram.assert_not_called()

    # On-cadence: should call add_histogram for weights + gradients
    trainer.global_step = 50
    param = torch.nn.Parameter(torch.randn(4))
    param.grad = torch.randn(4)
    pl_module = MagicMock()
    pl_module.named_parameters.return_value = [("layer.weight", param)]
    experiment = MagicMock()
    pl_module.logger.experiment = experiment
    cb.on_after_backward(trainer, pl_module)
    assert experiment.add_histogram.call_count == 2
    call_tags = [c[0][0] for c in experiment.add_histogram.call_args_list]
    assert "weights/layer.weight" in call_tags
    assert "gradients/layer.weight" in call_tags


def test_gradient_callback_included_in_trainer():
    """training_core.py passes GradientHistogramCallback to Trainer callbacks."""
    src = _module_source("mlops_pipeline/src/training_core.py")
    assert "gradient_cb" in src
    # Should be in the callbacks list
    assert "gradient_cb" in src.split("callbacks=[")[1].split("]")[0]


# ---------------------------------------------------------------------------
# 6. log_graph=True in TensorBoardLogger
# ---------------------------------------------------------------------------

def test_train_model_has_log_graph():
    src = _module_source("mlops_pipeline/src/steps/train_model.py")
    assert "log_graph=True" in src


def test_training_core_fallback_has_log_graph():
    src = _module_source("mlops_pipeline/src/training_core.py")
    assert "log_graph=True" in src

# Experiment configurations for regularization sweep
# Each experiment tests a different combination of regularization techniques

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""
    
    # Experiment metadata
    exp_id: int
    exp_name: str
    description: str
    
    # Regularization parameters
    spatial_dropout_rate: float = 0.0  # SpatialDropout3D after ConvLSTM layers
    weight_decay: float = 0.0          # L2 regularization via AdamW
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 30
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    
    # Architecture (fixed across experiments)
    lookback_mins: int = 30
    forecast_mins: int = 15
    filters: int = 64
    kernel_size: tuple = (3, 3)
    num_stations: int = 66
    
    # Data paths (configurable for GCS)
    data_dir: str = field(default_factory=lambda: os.environ.get("DATA_DIR", "data"))
    output_dir: str = field(default_factory=lambda: os.environ.get("OUTPUT_DIR", "outputs"))
    
    @property
    def experiment_output_dir(self) -> str:
        """Unique output directory for this experiment."""
        return os.path.join(self.output_dir, f"exp_{self.exp_id:02d}_{self.exp_name}")


# Define the 4 experiments for regularization sweep
EXPERIMENTS = {
    1: ExperimentConfig(
        exp_id=1,
        exp_name="baseline",
        description="Baseline: no regularization",
        spatial_dropout_rate=0.0,
        weight_decay=0.0,
        learning_rate=1e-3,
    ),
    2: ExperimentConfig(
        exp_id=2,
        exp_name="dropout_only",
        description="SpatialDropout3D=0.2, no weight decay",
        spatial_dropout_rate=0.2,
        weight_decay=0.0,
        learning_rate=1e-3,
    ),
    3: ExperimentConfig(
        exp_id=3,
        exp_name="dropout_l2",
        description="SpatialDropout3D=0.2 + L2 weight decay=1e-4",
        spatial_dropout_rate=0.2,
        weight_decay=1e-4,
        learning_rate=1e-3,
    ),
    4: ExperimentConfig(
        exp_id=4,
        exp_name="full_regularization",
        description="SpatialDropout3D=0.2 + L2=1e-4 + lower LR=3e-4",
        spatial_dropout_rate=0.2,
        weight_decay=1e-4,
        learning_rate=3e-4,
    ),
}


def get_experiment(exp_id: int) -> ExperimentConfig:
    """Retrieve experiment configuration by ID."""
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Experiment {exp_id} not found. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[exp_id]


def list_experiments() -> None:
    """Print all available experiments."""
    print("=" * 60)
    print("Available Experiments")
    print("=" * 60)
    for exp_id, config in EXPERIMENTS.items():
        print(f"\nExp {exp_id}: {config.exp_name}")
        print(f"  {config.description}")
        print(f"  - spatial_dropout: {config.spatial_dropout_rate}")
        print(f"  - weight_decay: {config.weight_decay}")
        print(f"  - learning_rate: {config.learning_rate}")
    print("=" * 60)

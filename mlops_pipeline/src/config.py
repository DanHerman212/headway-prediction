import os
import sys
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

def load_env_file(filepath: str):
    """Simple parser for .env files to populate os.environ"""
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Load .env if present (useful for local development)
# In vertex, these will be set by the container env
_msg = "Checking for .env file..."
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    print(f"{_msg} Found at {env_path}")
    load_env_file(env_path)
else:
    print(f"{_msg} Not found (relying on system environment)")

@dataclass
class Config:
    """
    Single source of truth for configuration.
    Reads purely from os.environ.
    """
    
    # GCP
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    region: str = os.getenv("VERTEX_LOCATION", "us-east1")
    artifact_bucket: str = os.getenv("ARTIFACT_BUCKET", "")
    data_lake_bucket: str = os.getenv("DATA_LAKE_BUCKET", "")
    tensorboard_resource: str = os.getenv("TENSORBOARD_RESOURCE_NAME", "")
    bq_table_name: str = os.getenv("BQ_TABLE_NAME", "headway_prediction.ml")
    
    # Model
    model_name: str = os.getenv("MODEL_NAME", "gru_model")
    gru_units: List[int] = field(default_factory=lambda: [
        int(x) for x in os.getenv("GRU_UNITS", "64,32").split(",")
    ])
    dropout_rate: float = float(os.getenv("DROPOUT_RATE", 0.2))
    lookback_steps: int = int(os.getenv("LOOKBACK_STEPS", 30))
    forecast_steps: int = int(os.getenv("FORECAST_STEPS", 1))
    num_routes: int = int(os.getenv("NUM_ROUTES", 3))
    route_ids: List[str] = field(default_factory=lambda: [
        x.strip() for x in os.getenv("ROUTE_IDS", "A,C,E").split(",")
    ])
    
    # Training
    epochs: int = int(os.getenv("EPOCHS", 20))
    batch_size: int = int(os.getenv("BATCH_SIZE", 32))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 0.001))
    optimizer: str = os.getenv("OPTIMIZER", "adam")
    loss_function: str = os.getenv("LOSS_FUNCTION", "huber")
    
    # Data
    train_split: float = float(os.getenv("TRAIN_SPLIT", 0.7))
    val_split: float = float(os.getenv("VAL_SPLIT", 0.15))
    test_split: float = float(os.getenv("TEST_SPLIT", 0.15))
    
    # Tracking
    experiment_name: str = os.getenv("EXPERIMENT_NAME", "default_experiment")
    run_name: str = os.getenv("RUN_NAME", "default_run")
    track_histograms: bool = os.getenv("TRACK_HISTOGRAMS", "false").lower() == "true"
    histogram_freq: int = int(os.getenv("HISTOGRAM_FREQ", 1))

    # Pipeline
    pipeline_root: str = os.getenv("PIPELINE_ROOT", "")
    tf_image_uri: str = os.getenv("TENSORFLOW_IMAGE_URI", "")
    serving_image_uri: str = os.getenv("SERVING_IMAGE_URI", "")

    def __post_init__(self):
        """Validation"""
        if not self.project_id and "pytest" not in sys.modules:
             print("Warning: GCP_PROJECT_ID not set.")
        if self.train_split + self.val_split + self.test_split != 1.0:
            print(f"Warning: Splits do not sum to 1.0 ({self.train_split + self.val_split + self.test_split})")

# Singleton instance
config = Config()

if __name__ == "__main__":
    # Debug: print loaded config
    import pprint
    pprint.pprint(config.__dict__)

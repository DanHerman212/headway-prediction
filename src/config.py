# Config class to centralize all hyperparameters for model architecture
# Paper reference: Usama & Koutsopoulos (2025) arXiv:2510.03121
import os  # helps override paths when switching between local tests and production
from dataclasses import dataclass 

@dataclass
class Config:
   # Model hyperparameters (Paper Table 1)
   LOOKBACK_MINS: int = 30   # Extended: full train journey (~45-60 min)
   FORECAST_MINS: int = 15   # Paper: 15 min forecast

   # Training (Paper Table 1)
   BATCH_SIZE: int = 128             # Scaled for A100
   EPOCHS: int = 100                 # Paper: 100
   LEARNING_RATE: float = 5e-4     # adapted for stability for 64 filters
   EARLY_STOPPING_PATIENCE: int = 10 # Stop if no improvement

   # Architecture (Paper Table 1)
   FILTERS: int = 64          # adapted for y-topology
   KERNEL_SIZE: tuple = (5, 1) # (5, 1) improved receptive field to see across the 6-bin zero buffer
   NUM_STATIONS: int = 75     # A-line: 66 distance bins (0.5mi each)
   NUM_SCHEDULE_CHANNELS: int = 4 # inwood, lefferst, far rockaway, rockaway park
   
   # Buffer zone indices (for optional masking for reference only
   # BUFFER_INDICES = [49, 50, 51, 52, 53, 54]  # 6 zero-bins between branches

   # Data splits (60/20/20)
   TRAIN_SPLIT: float = 0.6
   VAL_SPLIT: float = 0.2
   TEST_SPLIT: float = 0.2

   # Scaling
   MAX_HEADWAY: float = 30.0  # Minutes - for MinMaxScaler range

   # Data paths
   # Default to local 'data/' folder but allow override via env
   DATA_DIR: str = os.environ.get("DATA_DIR", "data")
   DATA_GCS_PATH: str = os.environ.get("DATA_GCS_PATH", "gs://st-convnet-training-configuration/headway-prediction/data")

   HEADWAY_FILE: str = "headway_matrix_topology.npy"
   SCHEDULE_FILE: str = "schedule_matrix_4terminal_planned.npy"

   @property
   def headway_path(self):
      return os.path.join(self.DATA_DIR, self.HEADWAY_FILE)
   
   @property
   def schedule_path(self):
      return os.path.join(self.DATA_DIR, self.SCHEDULE_FILE)
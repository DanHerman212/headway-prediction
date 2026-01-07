# Config class to centralize all hyperparameters for model architecture
# Paper reference: Usama & Koutsopoulos (2025) arXiv:2510.03121
import os  # helps override paths when switching between local tests and production
from dataclasses import dataclass 

@dataclass
class Config:
   # Model hyperparameters (Paper Table 1)
   LOOKBACK_MINS: int = 30   # Paper: 30 min history
   FORECAST_MINS: int = 15   # Paper: 15 min forecast

   # Training (Paper Table 1)
   BATCH_SIZE: int = 32              # Paper: 32
   EPOCHS: int = 100                 # Paper: 100
   LEARNING_RATE: float = 1e-3       # Paper: Adam default
   EARLY_STOPPING_PATIENCE: int = 50 # Paper: 50 epochs

   # Architecture (Paper Table 1)
   FILTERS: int = 32          # Paper: 32 (not 64)
   KERNEL_SIZE: tuple = (3, 3)
   NUM_STATIONS: int = 66     # A-line: 66 sequence_ids

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

   HEADWAY_FILE: str = "headway_matrix_full.npy"
   SCHEDULE_FILE: str = "schedule_matrix_full.npy"
   STATION_MAP_FILE: str = "a_line_station_distances.csv"

   @property
   def headway_path(self):
      return os.path.join(self.DATA_DIR, self.HEADWAY_FILE)
   
   @property
   def schedule_path(self):
      return os.path.join(self.DATA_DIR, self.SCHEDULE_FILE)
   
   @property
   def station_map_path(self):
      return os.path.join(self.DATA_DIR, self.STATION_MAP_FILE)
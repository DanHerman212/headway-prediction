# config class to centralize all hyperparameters for model architecture
import os # helps overide paths when switching between local tests and production
from dataclasses import dataclass 

@dataclass
class Config:
   # model hyperparameters
   LOOKBACK_MINS: int = 30
   FORECAST_MINS: int = 15

   # training
   BATCH_SIZE: int = 128
   EPOCHS: int = 30
   LEARNING_RATE: float = 1e-3

   # architecture
   FILTERS: int = 64
   KERNEL_SIZE: tuple = (3, 3)
   NUM_STATIONS: int = 66

   # data paths
   # default to local 'data/' folder but allow overide via .env
   DATA_DIR: str = os.environ.get("DATA_DIR", "data")

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
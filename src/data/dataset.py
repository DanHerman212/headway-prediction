import numpy as np
import tensorflow as tf
from src.config import Config

class SubwayDataGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.headway_data = None
        self.schedule_data = None
        self.scaler = None  # Optional: set externally for custom scaling

    def load_data(self, normalize=False, max_headway=30.0):
        """
        Loads data from disk.
        
        Args:
            normalize: If True, applies simple [0,1] normalization (divide by max_headway)
                      If False, returns raw data (use external scaler like RobustScaler)
            max_headway: Maximum headway value for normalization (default 30 minutes)
        """
        print(f"Loading data from {self.config.DATA_DIR}...")

        # load raw numpy arrays
        self.headway_data = np.load(self.config.headway_path).astype('float32')
        self.schedule_data = np.load(self.config.schedule_path).astype('float32')

        print(f"Headway Shape: {self.headway_data.shape}")
        print(f"Schedule Shape: {self.schedule_data.shape}")
        
        if normalize:
            print(f"Normalizing data to [0, 1] range (max={max_headway})")
            self.headway_data = np.clip(self.headway_data / max_headway, 0, 1)
            self.schedule_data = np.clip(self.schedule_data / max_headway, 0, 1)

    def make_dataset(self, start_index=0, end_index=None, shuffle=False):
        """
        creates a tf.dataDatset using the Index-Map pattern
        """
        lookback = self.config.LOOKBACK_MINS
        forecast = self.config.FORECAST_MINS
        batch_size = self.config.BATCH_SIZE
        total_window_size = lookback + forecast

        total_records = len(self.headway_data)

        # determine valid rande of indices 
        # we need enough room for (i + lookback + forecast)
        max_start_index = total_records - total_window_size

        if end_index is None:
            end_index = max_start_index
        else:
            end_index = min(end_index, max_start_index)

        print(f"Creating dataset from index {start_index} to {end_index}")

        # 1. convert to TF constants (moves to GPU if available)
        # force data to stay on cpu to prevent "GPU Ping pong"
        with tf.device('/cpu:0'):
            headway_tensor = tf.constant(self.headway_data)
            schedule_tensor = tf.constant(self.schedule_data)

        # 2. create index Dataset
        indices_ds = tf.data.Dataset.range(start_index, end_index)

        # 3. define slicing function
        def split_window(i):
            # input 1: headway history [t: t+30]
            past_headway = headway_tensor[i : i + lookback]

            # input 2: future schedule [t+30 : t+45]
            # note schedule aligns with the target window
            future_schedule = schedule_tensor[i + lookback : i + total_window_size]

            # Target: future headway [t+30 : t+45]
            target_headway = headway_tensor[i + lookback : i + total_window_size]

            return(
                {"headway_input": past_headway, "schedule_input": future_schedule},
                target_headway
            )
        
        # 4. map the slice function and batch
        ds = indices_ds.map(split_window, num_parallel_calls=tf.data.AUTOTUNE)

        
        # Optimization pipeline order:
        # Map -> Cache -> Shuffle -> Batch -> Prefetch
        # This prevents cache from locking shuffle order

        # 1. Cache in MEMORY (no filename = RAM cache)
        # Using disk cache ("subway_cache_temp") is SLOW on Google Drive!
        ds = ds.cache()  # ← Memory cache, not disk

        # 2. Shuffle: randomize cached examples every epoch
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        
        # 3. Batch and prefetch
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
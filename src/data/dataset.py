import numpy as np
import tensorflow as tf
from src.config import Config

class SubwayDataGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.headway_data = None
        self.schedule_data = None

    def load_data(self):
        """Loads from disk"""
        print(f"Loading data from {self.config.DATA_DIR}...")

        # load raw numpy arrays
        self.headway_data = np.load(self.config.headway_path).astype('float32')
        self.schedule_data = np.load(self.config.schedule_path).astype('float32')

        print(f"Headway Shape: {self.headway_data.shape}")
        print(f"Schedule Shape: {self.schedule_data.shape}")

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

        
        # Crucial Logic Change:To prevent the "Cache" from locking the Shuffle order (which would give you the exact same batches every epoch), the correct order must be:
        # Map -> Cache -> Shuffle -> Batch -> Prefetch.

        # optimization pipeline
        # 1 cache: load mapped data into RAM once (after 1st epoch)
        ds = ds.cache() 

        # 2 suffle: randomize the cached examples every epoch
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        
        # 3 batch and prefetch
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
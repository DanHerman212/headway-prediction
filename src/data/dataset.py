import numpy as np
import tensorflow as tf
from src.config import Config
from pathlib import Path

class SubwayDataGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.headway_data = None
        self.schedule_data = None
        self.temporal_data = None  # Cyclical temporal features
        self.scaler = None  # Optional: set externally for custom scaling

    def load_data(self, normalize=False, max_headway=30.0, load_temporal=True):
        """
        Loads data from disk.
        
        Args:
            normalize: If True, applies simple [0,1] normalization (divide by max_headway)
                      If False, returns raw data (use external scaler like RobustScaler)
            max_headway: Maximum headway value for normalization (default 30 minutes)
            load_temporal: If True, loads cyclical temporal features (hour/day sin/cos)
        """
        print(f"Loading data from {self.config.DATA_DIR}...")

        # load raw numpy arrays
        self.headway_data = np.load(self.config.headway_path).astype('float32')
        self.schedule_data = np.load(self.config.schedule_path).astype('float32')

        print(f"Headway Shape: {self.headway_data.shape}")
        print(f"Schedule Shape: {self.schedule_data.shape}")
        
        # Load temporal features if available
        if load_temporal:
            temporal_path = Path(self.config.DATA_DIR) / "temporal_features.npy"
            if temporal_path.exists():
                self.temporal_data = np.load(temporal_path).astype('float32')
                print(f"Temporal Features Shape: {self.temporal_data.shape}")
            else:
                print(f"⚠️  Temporal features not found at {temporal_path}")
                print("   Run: python -m src.data.temporal to generate")
                self.temporal_data = None
        
        if normalize:
            print(f"Normalizing data to [0, 1] range (max={max_headway})")
            self.headway_data = np.clip(self.headway_data / max_headway, 0, 1)
            self.schedule_data = np.clip(self.schedule_data / max_headway, 0, 1)

    def make_dataset(self, start_index=0, end_index=None, shuffle=False, include_temporal=True):
        """
        creates a tf.data.Dataset using the Index-Map pattern
        
        Args:
            start_index: Starting index for slicing
            end_index: Ending index for slicing
            shuffle: Whether to shuffle the dataset
            include_temporal: Whether to include cyclical temporal features
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
        
        # Check if temporal features are available
        use_temporal = include_temporal and self.temporal_data is not None
        if include_temporal and self.temporal_data is None:
            print("⚠️  Temporal features requested but not loaded. Proceeding without them.")

        # 1. convert to TF constants (moves to GPU if available)
        # force data to stay on cpu to prevent "GPU Ping pong"
        with tf.device('/cpu:0'):
            headway_tensor = tf.constant(self.headway_data)
            schedule_tensor = tf.constant(self.schedule_data)
            if use_temporal:
                temporal_tensor = tf.constant(self.temporal_data)

        # 2. create index Dataset
        indices_ds = tf.data.Dataset.range(start_index, end_index)

        # 3. define slicing function
        def split_window(i):
            # input 1: headway history [t: t+30]
            # Raw shape: (30, 66, 2, 1), squeeze to (30, 66, 2)
            past_headway = headway_tensor[i : i + lookback]
            past_headway = tf.squeeze(past_headway, axis=-1)  # Remove trailing channel dim

            # input 2: future schedule [t+30 : t+45]
            # Raw shape: (15, 4, 1), squeeze to (15, 4)
            #
            # OPTION A: Use future planned schedule (paper's "what-if" dispatcher approach)
            future_schedule = schedule_tensor[i + lookback : i + total_window_size]
            future_schedule = tf.squeeze(future_schedule, axis=-1)
            #
            # OPTION B: Use last-known terminal headway repeated (production-realistic)
            # Uncomment below and comment OPTION A to test production scenario
            # last_known = schedule_tensor[i + lookback - 1 : i + lookback]  # shape (1, 4, 1)
            # last_known = tf.squeeze(last_known, axis=-1)                    # shape (1, 4)
            # future_schedule = tf.repeat(last_known, forecast, axis=0)       # shape (15, 4)

            # Target: future headway [t+30 : t+45]
            # Raw shape: (15, 66, 2, 1), squeeze to (15, 66, 2)
            target_headway = headway_tensor[i + lookback : i + total_window_size]
            target_headway = tf.squeeze(target_headway, axis=-1)

            inputs = {"headway_input": past_headway, "schedule_input": future_schedule}
            
            # Add temporal features if available
            if use_temporal:
                # Temporal features for the lookback window: (30, 4)
                past_temporal = temporal_tensor[i : i + lookback]
                inputs["temporal_input"] = past_temporal
            
            return inputs, target_headway
        
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
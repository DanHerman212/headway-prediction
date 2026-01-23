"""Test dataset creation with simpler approach."""

import os
import numpy as np
import pandas as pd
from data import DataExtractor, DataPreprocessor
from training import Trainer
from config import ModelConfig
import tensorflow as tf

# Set environment
os.environ['GCP_PROJECT_ID'] = 'realtime-headway-prediction'

# Load config
config = ModelConfig.from_env()
config.lookback_steps = 20
config.batch_size = 64

print("Extracting and preprocessing data...")
extractor = DataExtractor(config)
df_raw = extractor.extract()
preprocessor = DataPreprocessor(config)
df_preprocessed = preprocessor.preprocess(df_raw)

print(f"Preprocessed data shape: {df_preprocessed.shape}")

# Load arrays
preprocessor.save(df_preprocessed, 'data/X.csv')
trainer = Trainer(config)
trainer.load_data('data/X.csv')

print(f"Loaded data:")
print(f"  input_x: {trainer.input_x.shape}")
print(f"  input_t: {trainer.input_t.shape}")
print(f"  input_r: {trainer.input_r.shape}")

# Prepare targets
input_t_reshaped = trainer.input_t.reshape(-1, 1)
targets_combined = np.column_stack([input_t_reshaped, trainer.input_r])

print(f"\nCombined targets shape: {targets_combined.shape}")

# Create just one dataset without map
print("\nCreating dataset without map...")
dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=trainer.input_x,
    targets=targets_combined,
    sequence_length=20,
    sampling_rate=1,
    batch_size=64,
    shuffle=False,
    start_index=20,
    end_index=1000
)

print("Dataset created successfully!")

# Get first batch
for x, y in dataset.take(1):
    print(f"\nBatch shapes:")
    print(f"  Inputs: {x.shape}")
    print(f"  Targets: {y.shape}")
    
print("\nTest complete!")

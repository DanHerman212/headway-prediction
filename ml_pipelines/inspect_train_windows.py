"""Inspect training dataset windows to verify correctness."""

import os
import numpy as np
import pandas as pd
from data import DataExtractor, DataPreprocessor
from training import Trainer
from config import ModelConfig

os.environ['GCP_PROJECT_ID'] = 'realtime-headway-prediction'

config = ModelConfig.from_env()
config.lookback_steps = 20
config.batch_size = 64

print("Loading and preprocessing data...")
extractor = DataExtractor(config)
df_raw = extractor.extract()
preprocessor = DataPreprocessor(config)
df_preprocessed = preprocessor.preprocess(df_raw)
preprocessor.save(df_preprocessed, 'data/X.csv')

print("Creating training dataset...")
trainer = Trainer(config)
trainer.load_data('data/X.csv')

# Create only train dataset
input_x_32 = trainer.input_x.astype(np.float32)
input_t_reshaped = trainer.input_t.reshape(-1, 1)
targets_combined = np.column_stack([input_t_reshaped, trainer.input_r]).astype(np.float32)

n = len(input_x_32)
train_end = int(n * 0.6)

import tensorflow as tf

def split_targets(x, y):
    return x, (y[:, 0:1], y[:, 1:4])

train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=input_x_32,
    targets=targets_combined,
    sequence_length=20,
    sampling_rate=1,
    batch_size=64,
    shuffle=False,  # No shuffle for inspection
    start_index=20,
    end_index=train_end
).map(split_targets)

print(f"\n{'='*70}")
print("TRAINING DATASET INSPECTION")
print(f"{'='*70}")

# Get first batch
for batch_inputs, (batch_headway, batch_routes) in train_dataset.take(1):
    print(f"\nBatch shapes:")
    print(f"  Inputs:  {batch_inputs.shape}")  # (64, 20, 8)
    print(f"  Headway: {batch_headway.shape}")  # (64, 1)
    print(f"  Routes:  {batch_routes.shape}")   # (64, 3)
    
    print(f"\n{'='*70}")
    print("FIRST 3 SAMPLES FROM BATCH")
    print(f"{'='*70}")
    
    for i in range(3):
        print(f"\n{'─'*70}")
        print(f"SAMPLE {i+1}")
        print(f"{'─'*70}")
        
        # The window for sample i uses data from indices [20+i-20 : 20+i] = [i : 20+i]
        # The target is at index 20+i
        sample_start_idx = i
        sample_end_idx = 20 + i
        target_idx = 20 + i
        
        print(f"\nWindow indices: [{sample_start_idx} : {sample_end_idx}]")
        print(f"Target index: {target_idx}")
        
        print(f"\nInput window shape: {batch_inputs[i].shape}")  # (20, 8)
        print(f"\nFirst 3 timesteps of window:")
        print(batch_inputs[i][:3].numpy())
        
        print(f"\nLast 3 timesteps of window:")
        print(batch_inputs[i][-3:].numpy())
        
        print(f"\nTarget headway (log): {batch_headway[i].numpy()}")
        print(f"Target route (one-hot): {batch_routes[i].numpy()}")
        
        route_idx = np.argmax(batch_routes[i].numpy())
        route_map = {0: 'A', 1: 'C', 2: 'E'}
        print(f"Target route decoded: {route_map[route_idx]}")

print(f"\n{'='*70}")
print("VERIFY AGAINST RAW DATA")
print(f"{'='*70}")

# Load CSV to compare
df = pd.read_csv('data/X.csv')

print("\nFirst sample should use rows [0:20] to predict row 20")
print("\nRows 0-2 from CSV (should match first 3 timesteps above):")
print(df.iloc[0:3].to_string())

print("\nRows 17-19 from CSV (should match last 3 timesteps above):")
print(df.iloc[17:20].to_string())

print("\nRow 20 from CSV (the target):")
print(df.iloc[20].to_string())

print(f"\n{'='*70}")
print("INSPECTION COMPLETE")
print(f"{'='*70}")

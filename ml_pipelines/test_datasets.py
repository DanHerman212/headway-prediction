"""Test dataset creation and inspect first batch."""

import os
import numpy as np
import pandas as pd
from data import DataExtractor, DataPreprocessor
from training import Trainer
from config import ModelConfig

# Set environment
os.environ['GCP_PROJECT_ID'] = 'realtime-headway-prediction'

# Load config
config = ModelConfig.from_env()
config.lookback_steps = 20
config.batch_size = 64

print("="*70)
print("STEP 1: Extract data from BigQuery")
print("="*70)
extractor = DataExtractor(config)
df_raw = extractor.extract()
extractor.save(df_raw, 'data/raw_data.csv')

print("\n" + "="*70)
print("STEP 2: Preprocess data")
print("="*70)
preprocessor = DataPreprocessor(config)
df_preprocessed = preprocessor.preprocess(df_raw)
preprocessor.save(df_preprocessed, 'data/X.csv')
print(f"Preprocessed data shape: {df_preprocessed.shape}")
print(f"  Columns: {list(df_preprocessed.columns)}")

print("\n" + "="*70)
print("STEP 3: Load and create datasets")
print("="*70)
trainer = Trainer(config)
trainer.load_data('data/X.csv')
train_dataset, val_dataset, test_dataset = trainer.create_datasets()

print("\n" + "="*70)
print("STEP 4: Inspect first batch")
print("="*70)

# Get first batch
for batch_inputs, batch_targets in train_dataset.take(1):
    print(f"\nBatch shapes:")
    print(f"  Inputs:  {batch_inputs.shape}")  # Should be (batch_size, 20, 8)
    print(f"  Target headway: {batch_targets[0].shape}")  # Should be (batch_size, 1)
    print(f"  Target route:   {batch_targets[1].shape}")  # Should be (batch_size, 3)
    
    print(f"\n{'='*70}")
    print("First 3 samples from batch:")
    print(f"{'='*70}")
    
    for i in range(3):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input window shape: {batch_inputs[i].shape}")  # (20, 8)
        print(f"Input window (first 3 timesteps):")
        print(batch_inputs[i][:3].numpy())
        
        print(f"\nTarget headway: {batch_targets[0][i].numpy()}")
        print(f"Target route (one-hot): {batch_targets[1][i].numpy()}")
        
        # Decode route
        route_idx = np.argmax(batch_targets[1][i].numpy())
        route_map = {0: 'A', 1: 'C', 2: 'E'}
        print(f"Target route decoded: {route_map[route_idx]}")

print(f"\n{'='*70}")
print("VERIFICATION: Compare to raw preprocessed data")
print(f"{'='*70}")

# Load preprocessed data to verify alignment
df = pd.read_csv('data/X.csv')
print(f"\nFirst 3 rows of preprocessed data (these should match first input window):")
print(df.head(3))

print(f"\nRow 20 (first target):")
print(df.iloc[20])

print("\nTest complete!")

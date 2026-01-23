"""Test index-based dataset creation."""

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

print("="*70)
print("STEP 1: Extract and preprocess data")
print("="*70)
extractor = DataExtractor(config)
df_raw = extractor.extract()
preprocessor = DataPreprocessor(config)
df_preprocessed = preprocessor.preprocess(df_raw)
preprocessor.save(df_preprocessed, 'data/X.csv')
print(f"Preprocessed shape: {df_preprocessed.shape}")

print("\n" + "="*70)
print("STEP 2: Create datasets (index-based)")
print("="*70)
trainer = Trainer(config)
trainer.load_data('data/X.csv')
train_dataset, val_dataset, test_dataset = trainer.create_datasets()

print("\n" + "="*70)
print("STEP 3: Inspect first batch from train dataset")
print("="*70)

for batch_inputs, (batch_headway, batch_routes) in train_dataset.take(1):
    print(f"\nBatch shapes:")
    print(f"  Inputs:  {batch_inputs.shape}")
    print(f"  Headway: {batch_headway.shape}")
    print(f"  Routes:  {batch_routes.shape}")
    
    print(f"\nFirst 3 samples:")
    for i in range(3):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input window shape: {batch_inputs[i].shape}")
        print(f"First timestep: {batch_inputs[i][0].numpy()}")
        print(f"Last timestep:  {batch_inputs[i][-1].numpy()}")
        print(f"Target headway: {batch_headway[i].numpy()}")
        print(f"Target route:   {batch_routes[i].numpy()}")
        
        route_idx = np.argmax(batch_routes[i].numpy())
        route_map = {0: 'A', 1: 'C', 2: 'E'}
        print(f"Route decoded: {route_map[route_idx]}")

print("\n" + "="*70)
print("SUCCESS: All three datasets created!")
print("="*70)

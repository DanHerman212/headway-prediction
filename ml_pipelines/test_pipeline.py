"""Test config-driven data pipeline."""

import os
from data import DataExtractor, DataPreprocessor
from config import ModelConfig

# Load config from environment (or use defaults)
config = ModelConfig.from_env()

print("Configuration:")
print(f"  Project: {config.bq_project}")
print(f"  Track: {config.track}")
print(f"  Routes: {config.route_ids}")
print(f"  Splits: {config.train_split}/{config.val_split}/{config.test_split}")

# Extract data
print("\nExtracting data...")
extractor = DataExtractor(config)
df = extractor.extract()
print(f"✓ Extracted {len(df):,} rows")

# Preprocess data
print("\nPreprocessing...")
preprocessor = DataPreprocessor(config)
train, val, test = preprocessor.preprocess(df)

print(f"✓ Train: {train.shape}")
print(f"✓ Val:   {val.shape}")
print(f"✓ Test:  {test.shape}")
print(f"\nFeatures: [log_headway, route_A, route_C, route_E, hour_sin, hour_cos, day_sin, day_cos]")

"""Test single dataset creation to isolate the issue."""

import os
import numpy as np
from data import DataExtractor, DataPreprocessor
from training import Trainer
from config import ModelConfig
import tensorflow as tf

os.environ['GCP_PROJECT_ID'] = 'realtime-headway-prediction'

config = ModelConfig.from_env()
config.lookback_steps = 20
config.batch_size = 64

print("Loading data...")
extractor = DataExtractor(config)
df_raw = extractor.extract()
preprocessor = DataPreprocessor(config)
df_preprocessed = preprocessor.preprocess(df_raw)
preprocessor.save(df_preprocessed, 'data/X.csv')

trainer = Trainer(config)
trainer.load_data('data/X.csv')

# Cast to float32
input_x_32 = trainer.input_x.astype(np.float32)
input_t_reshaped = trainer.input_t.reshape(-1, 1)
targets_combined = np.column_stack([input_t_reshaped, trainer.input_r]).astype(np.float32)

print(f"Data types: input_x={input_x_32.dtype}, targets={targets_combined.dtype}")

n = len(input_x_32)
train_end = int(n * 0.6)

print(f"\nCreating ONLY train dataset [{20} - {train_end}]...")

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=input_x_32,
    targets=targets_combined,
    sequence_length=20,
    batch_size=64,
    shuffle=True,
    start_index=20,
    end_index=train_end
)

print("Train dataset created!")

for x, y in train_ds.take(1):
    print(f"Batch shapes: x={x.shape}, y={y.shape}")

print("\nSuccess!")

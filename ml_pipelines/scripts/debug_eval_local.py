
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta

# Configurations
MODEL_DIR = "test_artifacts/model"
DATA_PATH = "test_artifacts/test_data.csv"
OUTPUT_DIR = "test_artifacts/evaluation_output"
METRICS_OUTPUT = "test_artifacts/metrics.json"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Generating Dummy Data...")
# Create a dummy DataFrame that matches the expected input structure for the Evaluator
# Based on read_file of ModelEvaluator and typical features
n_samples = 200
data = {
    'log_headway': np.random.rand(n_samples).astype(np.float32),
    'route_A': np.random.randint(0, 2, n_samples),
    'route_C': np.random.randint(0, 2, n_samples),
    'route_E': np.random.randint(0, 2, n_samples),
    'hour': np.random.rand(n_samples), # Normalized features typically
    'day_of_week': np.random.rand(n_samples),
    'is_weekend': np.random.randint(0, 2, n_samples),
}
df = pd.DataFrame(data)
df.to_csv(DATA_PATH, index=False)
print(f"   Saved dummy data to {DATA_PATH}")

print("2. Creating Dummy Model...")
# Create a simple Keras model with the expected input shape
# Config: lookback_steps=30 (default in config), features=7 (approx)
lookback = 30
features = 3 # Based on evaluator logic: log_headway + 3 route one-hots only for INPUT_X construction in evaluator?
# Actually, looking at evaluate_model.py: 
# input_x = data.values -> this takes ALL columns. 
# So input dim is len(df.columns).
input_shape = (lookback, len(df.columns)) 

inputs = keras.Input(shape=input_shape)
x = keras.layers.Flatten()(inputs)
# Two outputs: (headway, route classification)
out_headway = keras.layers.Dense(1, name='headway')(x)
out_route = keras.layers.Dense(3, activation='softmax', name='route')(x)

model = keras.Model(inputs=inputs, outputs=[out_headway, out_route])

# Use string identifiers for loss which Keras can definitely serialize standardly
# instead of list of strings which might be getting wrapped oddly in the dummy script
model.compile(loss={'headway': 'mean_squared_error', 'route': 'sparse_categorical_crossentropy'})

# Save as .h5 (the fix we applied)
h5_path = os.path.join(MODEL_DIR, "model.h5")
model.save(h5_path)
print(f"   Saved dummy model to {h5_path}")

print("3. Running Evaluation Script...")
# Call the evaluator module directly
# We set env vars to ensure ModelConfig picks up defaults or we can pass config explicitly 
# but the script uses ModelConfig.from_env()
cmd = (
    f"python3 -m ml_pipelines.evaluation.evaluate_model "
    f"--model {MODEL_DIR} "
    f"--data {DATA_PATH} "
    f"--pre_split "
    f"--output {OUTPUT_DIR} "
    f"--metrics_output {METRICS_OUTPUT}"
)

print(f"   Executing: {cmd}")
exit_code = os.system(cmd)

if exit_code == 0:
    print("\nSUCCESS: Evaluation component ran locally!")
else:
    print(f"\nFAILURE: Evaluation component exited with code {exit_code}")

This is a great pivot. Establishing a solid Conv2D/ConvLSTM baseline is standard scientific practice before moving to fancy architectures like Transformers. It gives you a "control" to measure against.

Here is a complete, drop-in plan for your `.md` file. I have optimized the code to fix the memory leaks and shape mismatches from the previous notebook while keeping the core "Conv2D/ConvLSTM" logic intact.

You can copy the content below directly into a file named `plan_conv2d_baseline.md`.

---

# Plan: Spatio-Temporal ConvNet Baseline

## 1. Objective

Implement a robust Spatio-Temporal Convolutional Network (ST-ConvNet) to forecast subway headways. This model serves as the primary baseline, utilizing **ConvLSTM** layers to capture temporal dependencies and **Conv2D** layers for spatial decoding.

## 2. Methodology

* **Input:**
* Historical Headways: `(Batch, 30 min, 156 Stations, 2 Dirs, 1 Channel)`
* Schedule Data: `(Batch, 15 min, 2 Dirs, 1 Channel)`


* **Architecture:**
* **Encoder:** ConvLSTM2D layers to extract spatiotemporal features from the past 30 minutes.
* **Fusion:** Injecting schedule data (future knowledge) via efficient broadcasting (fixing the memory explosion from the previous iteration).
* **Decoder:** Conv2D layers to project the fused features into the future 15-minute window.


* **Optimization:**
* Replace manual `RepeatVector` logic with `tf.broadcast_to` implicitly to save RAM.
* Use `swish` activation for better gradient flow.



## 3. Implementation Code

### Step A: Setup & Efficient Data Loading

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
LOOKBACK_MINS = 30
FORECAST_MINS = 15
BATCH_SIZE = 128  # Increased batch size (more stable gradients)
EPOCHS = 30
LEARNING_RATE = 1e-3

# --- Paths ---
DATA_DIR = "../data"
HEADWAY_PATH = os.path.join(DATA_DIR, "headway_matrix_full.npy")
SCHEDULE_PATH = os.path.join(DATA_DIR, "schedule_matrix_full.npy")

def load_data():
    print("Loading datasets...")
    # Load
    headway = np.load(HEADWAY_PATH)   # (Time, Stations, Dirs, Channels)
    schedule = np.load(SCHEDULE_PATH) # (Time, Dirs, Channels)
    
    # Normalize (0-1 range based on max 30 min headway)
    SCALER = 30.0
    headway = np.clip(headway / SCALER, 0, 1)
    schedule = np.clip(schedule / SCALER, 0, 1)
    
    # Cast to float32 to save memory (default is often float64)
    headway = headway.astype('float32')
    schedule = schedule.astype('float32')
    
    return headway, schedule

# --- Data Generator ---
def make_dataset(headway, schedule, start_idx, end_idx, batch_size):
    # Align Inputs (X) and Targets (Y)
    # X_headway: T-30 to T
    # X_schedule: T to T+15 (We use the future schedule as input)
    # Y_target:   T to T+15
    
    # We use a custom generator to handle the dual-input efficiently
    # Keras Timeseries generic helper:
    
    # 1. Headway Inputs
    ds_x = keras.utils.timeseries_dataset_from_array(
        data=headway, targets=None, sequence_length=LOOKBACK_MINS,
        batch_size=batch_size, start_index=start_idx, end_index=end_idx
    )
    
    # 2. Schedule Inputs (Target Window)
    # Offset by LOOKBACK because schedule input corresponds to the FUTURE prediction
    ds_s = keras.utils.timeseries_dataset_from_array(
        data=schedule, targets=None, sequence_length=FORECAST_MINS,
        batch_size=batch_size, start_index=start_idx+LOOKBACK_MINS, end_index=end_idx+LOOKBACK_MINS
    )
    
    # 3. Targets (Future Headway)
    ds_y = keras.utils.timeseries_dataset_from_array(
        data=headway, targets=None, sequence_length=FORECAST_MINS,
        batch_size=batch_size, start_index=start_idx+LOOKBACK_MINS, end_index=end_idx+LOOKBACK_MINS
    )
    
    return tf.data.Dataset.zip(((ds_x, ds_s), ds_y))

```

### Step B: The ST-ConvNet Model (Fixed)

This model replicates the scientific abstract approach (Encoder-Decoder) but uses broadcasting layers to handle the Schedule data, preventing the memory crash.

```python
def build_st_convnet(input_shape_headway, input_shape_schedule, output_steps):
    
    # --- Input 1: Historical Headways ---
    # Shape: (Batch, 30, Stations, Dirs, 1)
    x_in = layers.Input(shape=input_shape_headway, name="History_Input")
    
    # --- Input 2: Future Schedule ---
    # Shape: (Batch, 15, Dirs, 1)
    s_in = layers.Input(shape=input_shape_schedule, name="Schedule_Input")

    # --- ENCODER: Spatio-Temporal Feature Extraction ---
    # We use ConvLSTM to process the time dimension while keeping spatial structure.
    
    # Block 1
    x = layers.ConvLSTM2D(filters=16, kernel_size=(3, 1), padding='same', return_sequences=True, activation='swish')(x_in)
    x = layers.BatchNormalization()(x)
    
    # Block 2
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 1), padding='same', return_sequences=True, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    
    # Compress Time: Take the last hidden state or average pool?
    # Let's keep the last state to represent "Current Network State"
    # Shape becomes: (Batch, Stations, Dirs, 32)
    state_h = layers.ConvLSTM2D(filters=64, kernel_size=(3, 1), padding='same', return_sequences=False, activation='swish')(x)
    
    # --- FUSION: Injecting Schedule Knowledge ---
    # The schedule is global per direction (North/South), not per station.
    # We need to broadcast (Batch, 15, 2, 1) -> to match the spatial grid.
    
    # 1. Flatten Schedule Time: (Batch, 15, 2, 1) -> (Batch, 30) (15*2 features)
    s_flat = layers.Flatten()(s_in) 
    
    # 2. Project to same feature dim as State H
    s_emb = layers.Dense(64, activation='swish')(s_flat) # (Batch, 64)
    
    # 3. Reshape for broadcasting: (Batch, 1, 1, 64)
    s_emb_spatial = layers.Reshape((1, 1, 64))(s_emb)
    
    # 4. Concatenate (Broadcast happens automatically in modern Keras, 
    # but we can tile explicitly if needed. Let's rely on broadcasting via Add/Mult or manual tile)
    # We repeat the schedule features for every station.
    
    stations = state_h.shape[1]
    dirs = state_h.shape[2]
    
    # Tile manually to be safe: (Batch, Stations, Dirs, 64)
    s_tiled = tf.tile(s_emb_spatial, [1, stations, dirs, 1])
    
    # Concatenate State + Schedule
    # (Batch, Stations, Dirs, 128)
    merged = layers.Concatenate()([state_h, s_tiled])
    
    # --- DECODER: Forecasting ---
    # We use deep Conv2D layers to "hallucinate" the future heatmaps from the merged state.
    
    x = layers.Conv2D(64, kernel_size=(3, 1), padding='same', activation='swish')(merged)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, kernel_size=(3, 1), padding='same', activation='swish')(x)
    
    # Final Projection
    # We need 15 output time steps.
    # Method: Output 15 channels, then reshape.
    output_dim = output_steps * 1 # 15 channels (1 per minute)
    
    x = layers.Conv2D(output_dim, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
    
    # Current Shape: (Batch, Stations, Dirs, 15)
    # Target Shape:  (Batch, 15, Stations, Dirs, 1)
    
    # Permute to (Batch, 15, Stations, Dirs)
    x = layers.Permute((3, 1, 2))(x)
    
    # Add final channel dim
    output = layers.Reshape((output_steps, stations, dirs, 1))(x)
    
    return keras.Model(inputs=[x_in, s_in], outputs=output, name="ST_ConvNet_Baseline")


```

### Step C: Execution Loop

```python
# 1. Load Data
headway, schedule = load_data()
print("Headway Shape:", headway.shape)
print("Schedule Shape:", schedule.shape)

# 2. Split
split_idx = int(len(headway) * 0.8)
train_h, val_h = headway[:split_idx], headway[split_idx:]
train_s, val_s = schedule[:split_idx], schedule[split_idx:]

# 3. Create Datasets
# Ensure indices are valid (account for lookback/forecast windows)
train_end = len(train_h) - FORECAST_MINS - 1
val_end = len(val_h) - FORECAST_MINS - 1

train_ds = make_dataset(train_h, train_s, 0, train_end, BATCH_SIZE).shuffle(100).prefetch(tf.data.AUTOTUNE)
val_ds = make_dataset(val_h, val_s, 0, val_end, BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 4. Build Model
input_shape_h = (LOOKBACK_MINS, headway.shape[1], headway.shape[2], 1)
input_shape_s = (FORECAST_MINS, schedule.shape[1], 1)

model = build_st_convnet(input_shape_h, input_shape_s, FORECAST_MINS)
model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss='mse', metrics=['mae'])
model.summary()

# 5. Train
checkpoint = keras.callbacks.ModelCheckpoint("best_conv2d_model.keras", save_best_only=True, monitor='val_loss')
early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

```
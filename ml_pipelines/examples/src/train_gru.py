import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.cloud import storage
import joblib

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available and will be used.")
else:
    print("GPU is NOT available. Training will run on CPU.")

def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return tf.maximum(q * e, (q - 1) * e)

def combined_quantile_loss(y_true, y_pred):
    q1_pred = y_pred[:, 0:1]
    q5_pred = y_pred[:, 1:2]
    q9_pred = y_pred[:, 2:3]
    loss_q1 = tf.reduce_mean(quantile_loss(0.1, y_true, q1_pred))
    loss_q5 = tf.reduce_mean(quantile_loss(0.5, y_true, q5_pred))
    loss_q9 = tf.reduce_mean(quantile_loss(0.9, y_true, q9_pred))
    return loss_q1 + loss_q5 + loss_q9

def load_data_manual(input_path):
    print(f"Loading data from {input_path}...")
    with open(input_path) as f:
        data = f.read()

    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:]
    print(f"Header: {header}")
    print(f"Total lines: {len(lines)}")
    
    # Parse lines
    lines = [line for line in lines if line.strip()]

    mbt = np.zeros((len(lines),))
    # raw_data excludes the first column (date)
    raw_data = np.zeros((len(lines), len(header) - 1))
    
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        mbt[i] = values[1] # Assuming mbt is at index 1 of the values (2nd numeric column)
        raw_data[i, :] = values[:]
        
    return mbt, raw_data

def split_data(raw_data):
    n = len(raw_data)
    num_train_samples = int(0.6 * n)
    num_val_samples = int(0.2 * n)
    num_test_samples = n - num_train_samples - num_val_samples
    
    print("Training Samples:", num_train_samples)
    print("Validation Samples:", num_val_samples)
    print("Test Samples:", num_test_samples)
    
    return num_train_samples, num_val_samples, num_test_samples

def scale_data(raw_data, num_train_samples):
    train_mean = np.mean(raw_data[:num_train_samples, :], axis=0)
    train_std = np.std(raw_data[:num_train_samples, :], axis=0)
    
    # Apply standard scaling to all columns
    raw_data_scaled = (raw_data - train_mean) / train_std
    
    return raw_data_scaled, train_mean, train_std

def create_datasets(raw_data, mbt, num_train_samples, num_val_samples, sequence_length=160, batch_size=256):
    # Train Dataset
    train_dataset = keras.utils.timeseries_dataset_from_array(
        data=raw_data[:-sequence_length],
        targets=mbt[sequence_length:, None],
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=True,
        start_index=0,
        end_index=num_train_samples
    )
    
    # Validation Dataset
    val_dataset = keras.utils.timeseries_dataset_from_array(
        data=raw_data[:-sequence_length],
        targets=mbt[sequence_length:, None],
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=False,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples
    )
    
    # Test Dataset
    test_dataset = keras.utils.timeseries_dataset_from_array(
        data=raw_data[:-sequence_length],
        targets=mbt[sequence_length:, None],
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=False,
        start_index=num_train_samples + num_val_samples,
        end_index=None
    )
    
    return train_dataset, val_dataset, test_dataset

def run_experiment(model, train_ds, val_ds, model_name, epochs=50, patience=5):
    # 1. Optimization Schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=epochs * len(train_ds),
        alpha=0.001 
    )

    # 2. Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model_name}.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 3. Compile
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss=combined_quantile_loss,
        metrics=["mae"]
    )

    # 4. Fit
    print(f"\nStarting training for: {model_name}")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    return history

def build_bidirectional_gru_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Layer 1: Bidirectional GRU (Best Params: units=32, dropout=0.1)
    x = layers.Bidirectional(layers.GRU(32, return_sequences=True))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    
    # Layer 2: GRU (Best Params: units=32, dropout=0.4)
    x = layers.GRU(32, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Dense Head (3 outputs for quantiles 0.1, 0.5, 0.9)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="bidirectional_gru")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_dir', type=str, default='gru_model', help='Local directory to save model')
    parser.add_argument('--test_dataset_path', type=str, required=False, help='Path to save test dataset')
    args = parser.parse_args()

    # 1. Load Data
    mbt, raw_data = load_data_manual(args.input_csv)
    
    # 2. Split
    num_train, num_val, num_test = split_data(raw_data)
    
    # 3. Scale
    raw_data_scaled, train_mean, train_std = scale_data(raw_data, num_train)
    
    # 4. Create Datasets
    train_ds, val_ds, test_ds = create_datasets(raw_data_scaled, mbt, num_train, num_val)
    
    # Save Test Dataset if path provided
    if args.test_dataset_path:
        print(f"Saving test dataset to {args.test_dataset_path}...")
        os.makedirs(os.path.dirname(args.test_dataset_path), exist_ok=True)
        test_ds.save(args.test_dataset_path)
    
    # 5. Build Model
    input_shape = (160, raw_data_scaled.shape[1])
    model = build_bidirectional_gru_model(input_shape)
    model.summary()
    
    # 6. Run Experiment
    # Note: We use combined_quantile_loss instead of Huber
    # We also need to update the compile step in run_experiment, but since run_experiment
    # hardcodes the compile step, we should update run_experiment or do it here.
    # Let's update run_experiment to accept loss or just update it in place.
    
    # Actually, let's redefine run_experiment here or update the function above.
    # I will update the function above in a separate edit to be clean.
    history = run_experiment(model, train_ds, val_ds, "bidirectional_gru", epochs=50, patience=5)
    
    # 7. Save Model (SavedModel format for Vertex AI)
    best_model_path = "bidirectional_gru.keras"

    
    # Ensure output directory exists
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        best_model = keras.models.load_model(best_model_path, custom_objects={'combined_quantile_loss': combined_quantile_loss})
        
        print(f"Saving model to {args.model_dir} in SavedModel format...")
        best_model.export(args.model_dir)

        print(f"Saving model to {args.model_dir} in Keras format...")
        best_model.save(os.path.join(args.model_dir, "gru_model.keras"))
    else:
        print(f"Best model not found. Saving current model to {args.model_dir}...")
        print(f"Saving model to {args.model_dir} in SavedModel format...")
        model.export(args.model_dir)

        model.save(os.path.join(args.model_dir, "gru_model.keras"))

    # 8. Save Scaler (after model export to ensure directory exists and isn't overwritten)
    print(f"Saving scaler to {args.model_dir}...")
    # Ensure directory exists (export should have created it, but just in case)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    joblib.dump({'mean': train_mean, 'std': train_std}, os.path.join(args.model_dir, 'scaler.pkl'))

    # 9. Save Training History
    print(f"Saving training history to {args.model_dir}...")
    import json
    history_path = os.path.join(args.model_dir, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)



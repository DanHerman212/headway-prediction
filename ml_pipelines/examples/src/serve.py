import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
import joblib
from google.cloud import storage

app = Flask(__name__)

# Global variables
model = None
scaler = None
EXPECTED_SEQUENCE_LENGTH = 160
# Columns expected in the input (excluding 'ds' which is dropped)
# Based on mini_data.csv header: ds,duration,y,dow,temp,precip,snow,snowdepth,visibility,windspeed,rolling_std_50,rolling_max_10,rolling_mean_10,rolling_std_10,rolling_mean_50
# raw_data takes everything from index 1:
FEATURE_COLUMNS = [
    "duration", "y", "dow", "temp", "precip", "snow", "snowdepth", 
    "visibility", "windspeed", "rolling_std_50", "rolling_max_10", 
    "rolling_mean_10", "rolling_std_10", "rolling_mean_50"
]

def combined_quantile_loss(y_true, y_pred):
    # Needed for loading the model
    q1_pred = y_pred[:, 0:1]
    q5_pred = y_pred[:, 1:2]
    q9_pred = y_pred[:, 2:3]
    
    def quantile_loss(q, y_true, y_pred):
        e = y_true - y_pred
        return tf.maximum(q * e, (q - 1) * e)
        
    loss_q1 = tf.reduce_mean(quantile_loss(0.1, y_true, q1_pred))
    loss_q5 = tf.reduce_mean(quantile_loss(0.5, y_true, q5_pred))
    loss_q9 = tf.reduce_mean(quantile_loss(0.9, y_true, q9_pred))
    return loss_q1 + loss_q5 + loss_q9

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def load_artifacts():
    global model, scaler
    
    # Vertex AI mounts the model artifacts at AIP_STORAGE_URI or similar, 
    # but usually copies them to a local directory if using a pre-built container.
    # For custom containers, Vertex AI passes the GCS path in AIP_STORAGE_URI.
    # However, we can also assume the model is copied to the container or we download it.
    
    # When deploying a custom container, Vertex AI sets AIP_STORAGE_URI to the GCS path of the model artifact.
    # We need to download it.
    
    model_gcs_uri = os.environ.get("AIP_STORAGE_URI")
    if model_gcs_uri:
        print(f"AIP_STORAGE_URI found: {model_gcs_uri}")
        # Parse GCS URI
        if model_gcs_uri.startswith("gs://"):
            parts = model_gcs_uri[5:].split("/")
            bucket_name = parts[0]
            prefix = "/".join(parts[1:])
            
            # Download model file
            # We expect 'gru_model.keras' and 'scaler.pkl' in that directory
            os.makedirs("artifacts", exist_ok=True)
            
            try:
                download_blob(bucket_name, f"{prefix}/gru_model.keras", "artifacts/gru_model.keras")
                download_blob(bucket_name, f"{prefix}/scaler.pkl", "artifacts/scaler.pkl")
            except Exception as e:
                print(f"Error downloading artifacts: {e}")
                # Fallback for local testing
                pass
        
    # Load Model
    model_path = "artifacts/gru_model.keras"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path, custom_objects={'combined_quantile_loss': combined_quantile_loss})
    else:
        print("Model file not found!")

    # Load Scaler
    scaler_path = "artifacts/scaler.pkl"
    if os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
    else:
        print("Scaler file not found!")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    global model, scaler
    
    if not model or not scaler:
        # Try loading again (useful if before_first_request didn't fire in some setups)
        load_artifacts()
        if not model or not scaler:
            return jsonify({"error": "Model or Scaler not loaded"}), 500

    try:
        req_json = request.get_json()
        instances = req_json.get("instances", [])
        
        if not instances:
            return jsonify({"error": "No instances provided"}), 400
            
        # We expect 'instances' to be a list of data points.
        # Since this is a sequence model, we need to construct the sequence.
        # Scenario 1: User sends a list of 160+ objects -> We take the last 160.
        # Scenario 2: User sends a batch of sequences (List of Lists) -> We process batch.
        
        # Let's assume the input is a flat list of records (dictionaries) representing a time series history.
        # We will convert this to a DataFrame.
        
        df = pd.DataFrame(instances)
        
        # Ensure columns are present and in correct order
        # Fill missing with 0 or handle error
        data_matrix = []
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400
            data_matrix.append(df[col].values)
            
        # Shape: (num_features, num_samples) -> Transpose to (num_samples, num_features)
        data_matrix = np.array(data_matrix).T
        
        # Check length
        if len(data_matrix) < EXPECTED_SEQUENCE_LENGTH:
            return jsonify({"error": f"Not enough data points. Expected at least {EXPECTED_SEQUENCE_LENGTH}, got {len(data_matrix)}"}), 400
            
        # Take the last 160 points
        input_sequence = data_matrix[-EXPECTED_SEQUENCE_LENGTH:]
        
        # Scale
        # scaler is {'mean': ..., 'std': ...}
        mean = scaler['mean']
        std = scaler['std']
        
        input_sequence_scaled = (input_sequence - mean) / std
        
        # Reshape for model: (1, 160, features)
        input_tensor = np.expand_dims(input_sequence_scaled, axis=0)
        
        # Predict
        predictions = model.predict(input_tensor)
        
        # Predictions are [q10, q50, q90]
        # Return formatted response
        result = {
            "predictions": [
                {
                    "q10": float(predictions[0][0]),
                    "q50": float(predictions[0][1]),
                    "q90": float(predictions[0][2])
                }
            ]
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load artifacts on startup
load_artifacts()

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=8080)

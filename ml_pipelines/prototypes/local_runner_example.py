import os
import sys
import shutil
import pandas as pd
import numpy as np

# Add project root to path so we can import our actual modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # ../../
sys.path.insert(0, os.path.join(project_root, "ml_pipelines"))

from ml_pipelines.config import ModelConfig
from ml_pipelines.training.train import Trainer
from ml_pipelines.models.gru_model import StackedGRUModel
# IMPORTANT: Import non-Vertex tracker for local dev if available, 
# or mock the tracker to avoid API calls.
# For this prototype, we assume we just want to run the TRAINING LOGIC.

# ==============================================================================
# LOCAL RUNNER
# ==============================================================================
# Advantages:
# 1. Runs instantly on your laptop (Debugger friendly).
# 2. Uses local CSV files instead of BigQuery.
# 3. Mocks the Configuration to point to local paths.
# 4. Validates Tensorflow graph logic before you touch the cloud.
# ==============================================================================

def run_local_training():
    print("🚀 Starting Local Runner...")
    
    # 1. Setup Local Artifact Workspace
    local_dir = "local_run_artifacts"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir)
    
    # 2. Generate Dummy Data (or load sample.csv if you have one)
    print("📊 Generating dummy data...")
    dummy_data = {
        'log_headway': np.random.rand(100),
        'route_A': np.random.randint(0, 2, 100),
        'route_C': np.random.randint(0, 2, 100),
        'route_E': np.random.randint(0, 2, 100),
        # Add other expected columns by your preprocessor
    }
    df = pd.DataFrame(dummy_data)
    input_csv = os.path.join(local_dir, "input.csv")
    df.to_csv(input_csv, index=False)
    
    # 3. Create Configuration (Overriding Cloud Defaults)
    print("⚙️ configuring...")
    config = ModelConfig(
        model_name="local_debug_model",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        lookback_steps=5,  # Small for speed
        batch_size=2,      # Small for speed
        epochs=1           # Just one epoch to verify it runs
    )
    
    # 4. Initialize Components
    print("🏗️ Building model...")
    model_builder = StackedGRUModel(config)
    model = model_builder.create()
    
    trainer = Trainer(config)
    trainer.load_data(input_csv)
    
    # 5. Run Training (No Experiment Tracker needed for logic check)
    print("🏃 Training...")
    try:
        # We manually call model.fit because Trainer.train() might expect specific callbacks
        # or we can mock the callbacks list.
        train_ds, val_ds, _ = trainer.create_datasets()
        model.fit(train_ds, validation_data=val_ds, epochs=1)
        print("✅ Training logic validated!")
    except Exception as e:
        print(f"❌ Training failed locally: {e}")
        raise e
        
    # 6. Save Artifacts
    model_save_path = os.path.join(local_dir, "model.h5")
    model.save(model_save_path)
    print(f"💾 Model saved to {model_save_path}")

if __name__ == "__main__":
    run_local_training()

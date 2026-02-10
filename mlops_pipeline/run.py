import os
import sys
import argparse

# Ensure the project root is always on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops_pipeline.pipeline import headway_training_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the Headway Training Pipeline")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="gs://mlops-artifacts-realtime-headway-prediction/data/training_data.parquet",
        help="Path to the training data parquet file"
    )
    args, hydra_overrides = parser.parse_known_args()

    # Ensure absolute path for data
    if args.data_path.startswith("gs://"):
        data_path = args.data_path
    else:
        data_path = os.path.abspath(args.data_path)

    print(f"Launching pipeline with data from: {data_path}")
    if hydra_overrides:
        print(f"Applying Hydra overrides: {hydra_overrides}")
    
    headway_training_pipeline(
        data_path=data_path,
        hydra_overrides=hydra_overrides
    )

if __name__ == "__main__":
    main()
import os
import argparse
from mlops_pipeline.pipeline import headway_training_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the Headway Training Pipeline")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="local_artifacts/processed_data/training_data.parquet",
        help="Path to the training data parquet file"
    )
    args = parser.parse_args()

    # Ensure absolute path for data
    if args.data_path.startswith("gs://"):
        data_path = args.data_path
    else:
        data_path = os.path.abspath(args.data_path)

    # Run the pipeline
    print(f"Launching pipeline with data from: {data_path}")
    
    headway_training_pipeline(
        data_path=data_path
    )

if __name__ == "__main__":
    main()
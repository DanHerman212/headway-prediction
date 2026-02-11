import os
import sys
import argparse

# Ensure the project root is always on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops_pipeline.pipeline import headway_training_pipeline
from mlops_pipeline.hpo_pipeline import headway_hpo_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Headway Prediction Pipelines")
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["training", "hpo"], 
        default="training",
        help="Pipeline to run: 'training' (standard) or 'hpo' (hyperparameter optimization)"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="gs://mlops-artifacts-realtime-headway-prediction/data/training_data.parquet",
        help="Path to the training data parquet file"
    )
    
    parser.add_argument(
        "--use-vizier-params",
        action="store_true",
        default=False,
        help="Fetch best hyperparameters from the latest Vizier study "
             "and apply them via OmegaConf.update in the training step."
    )
    
    # Capture remaining args as Hydra overrides
    args, hydra_overrides = parser.parse_known_args()

    # Ensure absolute path for data
    if args.data_path.startswith("gs://"):
        data_path = args.data_path
    else:
        data_path = os.path.abspath(args.data_path)

    print(f"Launching {args.mode.upper()} pipeline with data from: {data_path}")
    if hydra_overrides:
        print(f"Applying Hydra overrides: {hydra_overrides}")
    if args.use_vizier_params:
        print("Vizier param injection ENABLED")
    
    if args.mode == "training":
        headway_training_pipeline(
            data_path=data_path,
            hydra_overrides=hydra_overrides,
            use_vizier_params=args.use_vizier_params,
        )
    elif args.mode == "hpo":
        headway_hpo_pipeline(
            data_path=data_path,
            hydra_overrides=hydra_overrides
        )

if __name__ == "__main__":
    main()
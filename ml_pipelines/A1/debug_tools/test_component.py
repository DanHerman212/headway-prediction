"""Debug tool to test individual pipeline components locally."""
import sys
from pathlib import Path
import argparse
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.artifact_utils import save_to_artifact, load_from_artifact
import numpy as np


def test_preprocess_component():
    """Test the preprocessing component in isolation."""
    print("\n" + "="*80)
    print("TESTING PREPROCESS COMPONENT")
    print("="*80 + "\n")
    
    try:
        from src.preprocess import preprocess_pipeline
        from src.config import config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "preprocessed_npy"
            artifact_dir.mkdir()
            
            print(f"Output artifact dir: {artifact_dir}")
            print(f"Querying: {config.BQ_PROJECT}.{config.BQ_DATASET}.{config.BQ_TABLE}\n")
            
            # Run preprocessing
            preprocess_pipeline(str(artifact_dir))
            
            # Verify output
            output_file = artifact_dir / "preprocessed_data.npy"
            if output_file.exists():
                data = np.load(output_file)
                print(f"\n✓ Preprocessing succeeded")
                print(f"  Output shape: {data.shape}")
                print(f"  Output dtype: {data.dtype}")
                return True
            else:
                print(f"\n✗ Preprocessing failed: output file not found")
                return False
                
    except Exception as e:
        print(f"\n✗ Preprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_component(preprocessed_data_path: str):
    """Test the training component with existing preprocessed data."""
    print("\n" + "="*80)
    print("TESTING TRAIN COMPONENT")
    print("="*80 + "\n")
    
    try:
        from src.train import train_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            
            print(f"Input data: {preprocessed_data_path}")
            print(f"Output model dir: {model_dir}\n")
            
            # Run training (will only do a few epochs in test mode)
            train_model(
                preprocessed_npy_path=preprocessed_data_path,
                model_output_path=str(model_dir)
            )
            
            print(f"\n✓ Training component test completed")
            return True
            
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test dataset creation with synthetic data."""
    print("\n" + "="*80)
    print("TESTING DATASET CREATION")
    print("="*80 + "\n")
    
    try:
        from src.train import create_timeseries_datasets
        from src.config import config
        
        # Create synthetic data
        n_samples = 1000
        n_features = 8
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        print(f"Synthetic data shape: {X.shape}")
        print(f"Creating train/val/test splits...\n")
        
        train_end = int(len(X) * config.TRAIN_SPLIT)
        val_end = int(len(X) * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        test_end = len(X)
        
        train_ds, val_ds, test_ds = create_timeseries_datasets(
            X=X,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end
        )
        
        # Verify batches
        print("\nVerifying batch consistency...")
        for ds_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
            batch_count = 0
            for inputs, targets in ds.take(3):
                input_size = inputs.shape[0]
                route_size = targets['route_output'].shape[0]
                headway_size = targets['headway_output'].shape[0]
                
                if input_size != route_size or input_size != headway_size:
                    print(f"✗ {ds_name} batch {batch_count}: size mismatch "
                          f"({input_size}, {route_size}, {headway_size})")
                    return False
                batch_count += 1
            
            print(f"  ✓ {ds_name}: {batch_count} batches validated")
        
        print(f"\n✓ Dataset creation test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test individual pipeline components")
    parser.add_argument("--component", choices=['preprocess', 'train', 'dataset', 'all'],
                       default='all', help="Component to test")
    parser.add_argument("--preprocessed-data", help="Path to preprocessed data for train test")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.component in ['dataset', 'all']:
        results['dataset'] = test_dataset_creation()
    
    if args.component in ['preprocess', 'all']:
        results['preprocess'] = test_preprocess_component()
    
    if args.component == 'train':
        if not args.preprocessed_data:
            print("ERROR: --preprocessed-data required for train test")
            sys.exit(1)
        results['train'] = test_train_component(args.preprocessed_data)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {component:15s}: {status}")
    print("="*80 + "\n")
    
    # Exit with error if any test failed
    if not all(results.values()):
        sys.exit(1)

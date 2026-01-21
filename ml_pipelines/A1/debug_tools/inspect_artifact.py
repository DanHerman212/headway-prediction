"""Debugging tools for inspecting pipeline artifacts and state."""
import sys
from pathlib import Path
import numpy as np
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.artifact_utils import load_from_artifact, validate_preprocessed_data


def inspect_artifact(artifact_path: str, filename: str = "preprocessed_data.npy"):
    """Inspect a numpy artifact from GCS or local path.
    
    Usage:
        python inspect_artifact.py /path/to/artifact_dir
        python inspect_artifact.py gs://bucket/path/to/artifact_dir
    """
    print(f"\n{'='*80}")
    print(f"ARTIFACT INSPECTION: {artifact_path}")
    print(f"{'='*80}\n")
    
    try:
        # Load artifact
        print(f"Loading from: {Path(artifact_path) / filename}")
        data = load_from_artifact(artifact_path, filename)
        
        # Basic info
        print(f"\nBasic Information:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Size: {data.nbytes / (1024**2):.2f} MB")
        
        # Statistical summary
        print(f"\nStatistical Summary:")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std: {data.std():.4f}")
        
        # Feature-wise summary
        if data.ndim == 2 and data.shape[1] == 8:
            feature_names = [
                'log_headway', 'route_A', 'route_C', 'route_E',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
            ]
            
            print(f"\nFeature-wise Summary:")
            for i, name in enumerate(feature_names):
                feature_data = data[:, i]
                print(f"  {name:12s}: min={feature_data.min():7.3f}, "
                      f"max={feature_data.max():7.3f}, "
                      f"mean={feature_data.mean():7.3f}, "
                      f"std={feature_data.std():7.3f}")
        
        # Data quality checks
        print(f"\nData Quality:")
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        print(f"  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")
        
        # Validate if preprocessed data
        print(f"\nValidation:")
        is_valid, errors = validate_preprocessed_data(data)
        if is_valid:
            print("  ✓ All validation checks passed")
        else:
            print(f"  ✗ {len(errors)} validation errors:")
            for error in errors:
                print(f"    - {error}")
        
        # Sample data
        print(f"\nFirst 3 samples:")
        print(data[:3])
        
        print(f"\n{'='*80}\n")
        return data
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_artifacts(path1: str, path2: str, filename: str = "preprocessed_data.npy"):
    """Compare two artifacts to identify differences."""
    print(f"\n{'='*80}")
    print(f"COMPARING ARTIFACTS")
    print(f"{'='*80}\n")
    
    try:
        data1 = load_from_artifact(path1, filename)
        data2 = load_from_artifact(path2, filename)
        
        print(f"Artifact 1: {path1}")
        print(f"  Shape: {data1.shape}")
        print(f"\nArtifact 2: {path2}")
        print(f"  Shape: {data2.shape}")
        
        # Compare shapes
        if data1.shape != data2.shape:
            print(f"\n✗ Shape mismatch: {data1.shape} vs {data2.shape}")
            return
        
        # Compare values
        if np.array_equal(data1, data2):
            print(f"\n✓ Artifacts are identical")
        else:
            diff = np.abs(data1 - data2)
            print(f"\n✗ Artifacts differ:")
            print(f"  Max difference: {diff.max():.6f}")
            print(f"  Mean difference: {diff.mean():.6f}")
            print(f"  Differing elements: {(diff > 1e-6).sum()} / {data1.size}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect pipeline artifacts")
    parser.add_argument("artifact_path", help="Path to artifact directory")
    parser.add_argument("--filename", default="preprocessed_data.npy", help="Filename inside artifact directory")
    parser.add_argument("--compare", help="Path to second artifact for comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_artifacts(args.artifact_path, args.compare, args.filename)
    else:
        inspect_artifact(args.artifact_path, args.filename)

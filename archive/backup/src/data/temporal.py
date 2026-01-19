"""
Cyclical Temporal Encoding for Headway Prediction

Encodes time-of-day and day-of-week as sin/cos pairs to provide
the model with continuous, differentiable temporal context.

Reference: Deep Research recommendations for transit forecasting
"""
import numpy as np
import pandas as pd
from pathlib import Path


def create_cyclical_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert timestamps to cyclical sin/cos encodings.
    
    Args:
        timestamps: DatetimeIndex aligned with headway matrix rows
        
    Returns:
        np.ndarray of shape (T, 4) with columns:
            [hour_sin, hour_cos, day_sin, day_cos]
    """
    hours = timestamps.hour + timestamps.minute / 60.0  # Fractional hours
    days = timestamps.dayofweek  # 0=Monday, 6=Sunday
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    day_sin = np.sin(2 * np.pi * days / 7.0)
    day_cos = np.cos(2 * np.pi * days / 7.0)
    
    # Stack into (T, 4) array
    temporal_features = np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)
    
    return temporal_features.astype('float32')


def create_temporal_matrix_from_headway(
    headway_path: str,
    start_time: str = "2025-06-06 00:00:00",
    freq: str = "1min",
    tz: str = "UTC"
) -> np.ndarray:
    """
    Create temporal features matrix aligned with existing headway matrix.
    
    Since we don't have the original time index saved, we reconstruct it
    from the known start time and 1-minute frequency.
    
    Args:
        headway_path: Path to headway_matrix_full.npy
        start_time: Start timestamp of the headway matrix
        freq: Time frequency (default: 1 minute)
        tz: Timezone
        
    Returns:
        np.ndarray of shape (T, 4)
    """
    # Load headway to get number of timesteps
    headway = np.load(headway_path)
    num_timesteps = headway.shape[0]
    
    # Reconstruct time index
    timestamps = pd.date_range(
        start=start_time,
        periods=num_timesteps,
        freq=freq,
        tz=tz
    )
    
    print(f"Reconstructed time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Total timesteps: {num_timesteps}")
    
    return create_cyclical_features(timestamps)


def save_temporal_matrix(data_dir: str, start_time: str = "2025-06-06 00:00:00"):
    """
    Generate and save temporal features matrix to disk.
    
    Args:
        data_dir: Directory containing headway_matrix_full.npy
        start_time: Start timestamp of the data
    """
    data_path = Path(data_dir)
    headway_path = data_path / "headway_matrix_full.npy"
    output_path = data_path / "temporal_features.npy"
    
    temporal = create_temporal_matrix_from_headway(
        str(headway_path),
        start_time=start_time
    )
    
    np.save(output_path, temporal)
    print(f"Saved temporal features to {output_path}")
    print(f"Shape: {temporal.shape}")
    
    # Quick validation
    print("\nSample values (first 5 timesteps):")
    print("hour_sin | hour_cos | day_sin | day_cos")
    for i in range(5):
        print(f"{temporal[i, 0]:8.4f} | {temporal[i, 1]:8.4f} | {temporal[i, 2]:8.4f} | {temporal[i, 3]:8.4f}")
    
    return temporal


if __name__ == "__main__":
    # Generate temporal features for the A-line dataset
    save_temporal_matrix("data/", start_time="2025-06-06 00:00:00")

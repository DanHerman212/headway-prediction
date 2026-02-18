import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from typing import Tuple

def clean_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Apply physics-based cleaning, imputation, and time indexing.

    Returns
    -------
    (df, time_anchor_iso)
        Cleaned DataFrame and the ISO-8601 string of the global minimum
        arrival_time_dt, so downstream consumers can convert time_idx
        back to real timestamps.
    """
    df = data.copy()

    # 1. Ensure Categoricals are Strings
    cat_cols = ['group_id', 'route_id', 'direction', 'regime_id', 'track_id', 'preceding_route_id']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    # 2. Parse Dates
    if 'arrival_time' in df.columns:
        df['arrival_time_dt'] = pd.to_datetime(df['arrival_time'])

    # 3. Imputation Logic
    # Fill gaps/upstream with median
    for col in ['preceding_train_gap', 'upstream_headway_14th', 'travel_time_14th', 'travel_time_34th']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill deviations with 0.0 (assume on-time if unknown)
    dev_cols = [c for c in df.columns if 'deviation' in c]
    for col in dev_cols:
        df[col] = df[col].fillna(0.0)

    # 4. Handle 23rd St Express/Local Logic
    # stops_at_23rd and imputed travel times are computed upstream in generate_dataset.py
    # to match the streaming pipeline exactly. No recomputation needed here.
    if 'travel_time_23rd' in df.columns:
        df['travel_time_23rd'] = df['travel_time_23rd'].fillna(df['travel_time_23rd'].median())

    # 5. Correct Time Index (Physical Time)
    # Calculate absolute minutes elapsed since the global minimum time
    min_time = df['arrival_time_dt'].min()
    df['time_idx'] = ((df['arrival_time_dt'] - min_time).dt.total_seconds() / 60).astype(int)

    # Sort final dataframe
    df = df.sort_values(['group_id', 'time_idx'])

    # Return the anchor so callers can map time_idx → real timestamps
    time_anchor_iso = min_time.isoformat()
    
    return df, time_anchor_iso

def get_slice_with_lookback(full_df: pd.DataFrame, start_date, end_date, lookback: int):
    """
    Extracts a time slice from the dataframe but prepends 'lookback' steps
    from the history so the first prediction has context.
    """
    # Core slice
    mask = (full_df['arrival_time_dt'] >= start_date) & (full_df['arrival_time_dt'] < end_date)
    core_df = full_df[mask]

    # Context slice (history)
    prior_df = full_df[full_df['arrival_time_dt'] < start_date]
    pre_data = []

    for g_id, group in prior_df.groupby('group_id'):
        pre_data.append(group.tail(lookback))

    if pre_data:
        lookback_df = pd.concat(pre_data)
        return pd.concat([lookback_df, core_df]).sort_values(['group_id', 'time_idx'])
    
    return core_df

def create_datasets(data: pd.DataFrame, config: DictConfig, time_anchor_iso: str = ""):
    """
    Splits the data and initializes TimeSeriesDataSet objects based on the configuration.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned DataFrame from ``clean_dataset``.
    config : DictConfig
        Processing section of the Hydra config.
    time_anchor_iso : str
        ISO-8601 timestamp of the global min arrival_time_dt (time_idx=0).
    """
    # Parse cutoffs (handle timestamps safely)
    is_tz_aware = data['arrival_time_dt'].dt.tz is not None
    tz = "UTC" if is_tz_aware else None
    
    train_end = pd.Timestamp(config.train_end_date, tz=tz)
    val_end = pd.Timestamp(config.val_end_date, tz=tz)
    test_end = pd.Timestamp(config.test_end_date, tz=tz)

    # Create Physical Splits
    train_df = data[data['arrival_time_dt'] < train_end]
    val_df_input = get_slice_with_lookback(data, train_end, val_end, lookback=config.max_encoder_length)
    test_df_input = get_slice_with_lookback(data, val_end, test_end, lookback=config.max_encoder_length)

    # 1. Training Dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx=config.time_idx,
        target=config.target,
        group_ids=list(config.group_ids),
        min_encoder_length=config.min_encoder_length,
        max_encoder_length=config.max_encoder_length,
        min_prediction_length=config.min_prediction_length,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=list(config.static_categoricals),
        time_varying_known_categoricals=list(config.time_varying_known_categoricals),
        time_varying_known_reals=list(config.time_varying_known_reals),
        time_varying_unknown_categoricals=list(config.time_varying_unknown_categoricals),
        time_varying_unknown_reals=list(config.time_varying_unknown_reals),
        target_normalizer=GroupNormalizer(
            groups=list(config.group_ids), transformation=config.target_normalizer
        ),
        add_relative_time_idx=config.add_relative_time_idx,
        add_target_scales=config.add_target_scales,
        add_encoder_length=config.add_encoder_length,
        allow_missing_timesteps=True  # HARDCODED: Physical time index requires this
    )

    # 2. Validation & Test (from_dataset ensures consistent encoders)
    validation = TimeSeriesDataSet.from_dataset(training, val_df_input, predict=False, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df_input, predict=False, stop_randomization=True)

    return training, validation, test, time_anchor_iso
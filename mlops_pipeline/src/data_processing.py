import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import EncoderNormalizer
from typing import Dict, Tuple

def clean_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply physics-based cleaning and imputation.

    Returns
    -------
    (df, time_lookup)
        Cleaned DataFrame (with sequential time_idx from upstream parquet)
        and a lookup table mapping (group_id, time_idx) → arrival_time_dt
        so the evaluation module can reconstruct real timestamps.
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

    # 3b. Clip outliers — overnight gaps can exceed 5000 min, cap at 20
    # 98th percentile ≈ 20 min; beyond that = "no recent preceding train"
    if 'preceding_train_gap' in df.columns:
        df['preceding_train_gap'] = df['preceding_train_gap'].clip(upper=20.0)

    # Fill deviations with 0.0 (assume on-time if unknown)
    dev_cols = [c for c in df.columns if 'deviation' in c]
    for col in dev_cols:
        df[col] = df[col].fillna(0.0)

    # 4. Handle 23rd St Express/Local Logic
    # stops_at_23rd and imputed travel times are computed upstream in generate_dataset.py
    # to match the streaming pipeline exactly. No recomputation needed here.
    if 'travel_time_23rd' in df.columns:
        df['travel_time_23rd'] = df['travel_time_23rd'].fillna(df['travel_time_23rd'].median())

    # 5. Time Index — imported from parquet (sequential per group).
    #    No computation here; the upstream batch pipeline
    #    (ReindexTimeInGroupsFn) already assigned 0, 1, 2... per group_id.

    # Sort final dataframe
    df = df.sort_values(['group_id', 'time_idx'])

    # 6. Clip extreme headway outliers — p99 ≈ 29 min.
    #    Values beyond this are service disruptions / overnight gaps that
    #    disproportionately inflate MAE.  Cap at 30 min.
    HEADWAY_CAP = 30.0
    if 'service_headway' in df.columns:
        df['service_headway'] = df['service_headway'].clip(upper=HEADWAY_CAP)

    # 7. Lag and rolling features derived from service_headway.
    #    shift(1) ensures only *past* values are used (no leakage).
    g = df.groupby('group_id')['service_headway']
    df['headway_lag_1'] = g.shift(1)
    df['rolling_mean_5'] = g.shift(1).rolling(window=5, min_periods=3).mean().reset_index(level=0, drop=True)
    df['rolling_std_5']  = g.shift(1).rolling(window=5, min_periods=3).std().reset_index(level=0, drop=True)

    # 8. Drop rows where lag/rolling are NaN (first few per group).
    #    Then re-index time_idx sequentially so there are no gaps.
    n_before = len(df)
    df = df.dropna(subset=['headway_lag_1', 'rolling_mean_5'])
    df['time_idx'] = df.groupby('group_id').cumcount()
    n_after = len(df)
    import logging
    logging.getLogger(__name__).info(
        "Lag/rolling feature creation: dropped %d rows (%.1f%%), %d remain",
        n_before - n_after, 100 * (n_before - n_after) / n_before, n_after,
    )

    # Fill any remaining NaN in rolling_std (groups with < 2 valid values)
    df['rolling_std_5'] = df['rolling_std_5'].fillna(0.0)

    # Build lookup table: (group_id, time_idx) → arrival_time_dt
    time_lookup = df[['group_id', 'time_idx', 'arrival_time_dt']].copy()

    return df, time_lookup

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

def create_datasets(data: pd.DataFrame, config: DictConfig):
    """
    Splits the data and initializes TimeSeriesDataSet objects based on the configuration.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned DataFrame from ``clean_dataset``.
    config : DictConfig
        Processing section of the Hydra config.
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

    # Build per-feature scalers dict from config (if present)
    scalers_cfg: Dict[str, str] = dict(config.get("scalers", {}))
    scalers_map: Dict[str, EncoderNormalizer] = {}
    for col_name, method in scalers_cfg.items():
        scalers_map[col_name] = EncoderNormalizer(method=method)

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
        scalers=scalers_map if scalers_map else None,
        add_relative_time_idx=config.add_relative_time_idx,
        add_target_scales=config.add_target_scales,
        add_encoder_length=config.add_encoder_length,
        allow_missing_timesteps=True  # HARDCODED: Physical time index requires this
    )

    # 2. Validation & Test (from_dataset ensures consistent encoders)
    validation = TimeSeriesDataSet.from_dataset(training, val_df_input, predict=False, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df_input, predict=False, stop_randomization=True)

    return training, validation, test
import pandas as pd
from typing import Tuple
from pytorch_forecasting import TimeSeriesDataSet
from omegaconf import DictConfig

from ..data_processing import clean_dataset, create_datasets


def process_data(
    raw_data: pd.DataFrame,
    config: DictConfig,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    """
    Cleans the data and creates the TimeSeriesDataSet splits.

    Returns the three dataset splits plus a time_lookup DataFrame
    mapping (group_id, time_idx) → arrival_time_dt so the evaluation
    step can reconstruct human-readable timestamps.
    """
    # 1. Clean physics/imputation
    cleaned_df, time_lookup = clean_dataset(raw_data)

    # 2. Create splits based on config cutoffs
    training, validation, test = create_datasets(
        cleaned_df, config.processing
    )

    return training, validation, test, time_lookup
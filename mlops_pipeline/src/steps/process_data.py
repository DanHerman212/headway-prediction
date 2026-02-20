import pandas as pd
from typing import Tuple
from pytorch_forecasting import TimeSeriesDataSet
from omegaconf import DictConfig

from ..data_processing import clean_dataset, create_datasets


def process_data(
    raw_data: pd.DataFrame,
    config: DictConfig,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, str]:
    """
    Cleans the data and creates the TimeSeriesDataSet splits.

    Returns the three dataset splits plus the time anchor (ISO-8601 string
    of global min arrival_time_dt) so the evaluation step can map time_idx
    values back to human-readable timestamps.
    """
    # 1. Clean physics/imputation
    cleaned_df, time_anchor_iso = clean_dataset(raw_data)

    # 2. Create splits based on config cutoffs
    training, validation, test, time_anchor_iso = create_datasets(
        cleaned_df, config.processing, time_anchor_iso
    )

    return training, validation, test, time_anchor_iso
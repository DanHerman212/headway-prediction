import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
from pytorch_forecasting import TimeSeriesDataSet
from omegaconf import DictConfig

from ..data_processing import clean_dataset, create_datasets

@step
def process_data_step(
    raw_data: pd.DataFrame, 
    config: DictConfig
) -> Tuple[
    Annotated[TimeSeriesDataSet, "training_dataset"],
    Annotated[TimeSeriesDataSet, "validation_dataset"],
    Annotated[TimeSeriesDataSet, "test_dataset"]
]:
    """
    Cleans the data and creates the TimeSeriesDataSet splits.
    """
    # 1. Clean physics/imputation
    cleaned_df = clean_dataset(raw_data)
    
    # 2. Create splits based on config cutoffs
    training, validation, test = create_datasets(cleaned_df, config.processing)
    
    return training, validation, test
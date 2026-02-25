import pandas as pd


def ingest_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw data from a generic file path (local or cloud).
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to ingest data from {file_path}") from e
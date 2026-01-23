"""
BigQuery ETL Pipeline

Handles data extraction from BigQuery, transformation, and loading
for ML training pipelines.
"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class BigQueryETL:
    """
    ETL pipeline for extracting data from BigQuery and preparing it for ML training.
    
    Supports:
    - SQL query execution
    - Data validation
    - Train/val/test splitting
    - Feature scaling
    - Export to various formats (numpy, tf.data.Dataset)
    
    Example:
        etl = BigQueryETL(
            project_id="my-project",
            dataset_id="ml_data",
            table_id="training_data"
        )
        
        # Load data
        df = etl.load_data()
        
        # Split and scale
        train, val, test, scaler = etl.split_and_scale(
            df,
            splits=(0.6, 0.2, 0.2),
            scaling_method="minmax"
        )
        
        # Export to numpy
        X_train, y_train = etl.to_numpy(train, target_col="target")
    """
    
    def __init__(
        self,
        project_id: str,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        credentials: Optional[Any] = None
    ):
        """
        Initialize BigQuery ETL pipeline.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            credentials: Optional GCP credentials
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        
        # Initialize BigQuery client
        self.client = bigquery.Client(
            project=project_id,
            credentials=credentials
        )
        
        print(f"✓ BigQuery ETL initialized")
        print(f"  Project: {project_id}")
        if dataset_id and table_id:
            print(f"  Table: {dataset_id}.{table_id}")
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_data(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        **query_params
    ) -> pd.DataFrame:
        """
        Load data from BigQuery using SQL query or table reference.
        
        Args:
            query: Optional SQL query. If None, loads entire table.
            limit: Optional row limit
            **query_params: Parameters for parameterized queries
            
        Returns:
            DataFrame with query results
        """
        if query is None:
            # Load entire table
            if not self.dataset_id or not self.table_id:
                raise ValueError("Must provide dataset_id and table_id or a query")
            
            query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`"
            
            if limit:
                query += f" LIMIT {limit}"
        
        print(f"Executing query...")
        print(f"  Query: {query[:100]}...")
        
        # Execute query
        df = self.client.query(query, job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING", v)
                for k, v in query_params.items()
            ]
        )).to_dataframe()
        
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def load_from_table(
        self,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data directly from a BigQuery table.
        
        Args:
            dataset_id: Dataset ID (uses instance default if None)
            table_id: Table ID (uses instance default if None)
            limit: Optional row limit
            
        Returns:
            DataFrame with table data
        """
        dataset = dataset_id or self.dataset_id
        table = table_id or self.table_id
        
        if not dataset or not table:
            raise ValueError("Must provide dataset_id and table_id")
        
        query = f"SELECT * FROM `{self.project_id}.{dataset}.{table}`"
        if limit:
            query += f" LIMIT {limit}"
        
        return self.load_data(query=query)
    
    # =========================================================================
    # Data Transformation
    # =========================================================================
    
    def split_data(
        self,
        df: pd.DataFrame,
        splits: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        shuffle: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            splits: (train, val, test) split ratios
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility
            
        Returns:
            (train_df, val_df, test_df) tuple
        """
        train_ratio, val_ratio, test_ratio = splits
        
        # Validate splits
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Shuffle if requested
        if shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"✓ Data split:")
        print(f"  Train: {len(train_df):,} rows ({train_ratio:.1%})")
        print(f"  Val:   {len(val_df):,} rows ({val_ratio:.1%})")
        print(f"  Test:  {len(test_df):,} rows ({test_ratio:.1%})")
        
        return train_df, val_df, test_df
    
    def create_scaler(
        self,
        scaling_method: str = "minmax",
        feature_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Create a sklearn scaler based on scaling method.
        
        Args:
            scaling_method: "minmax", "standard", "robust", or "none"
            feature_range: Range for MinMaxScaler
            
        Returns:
            Scaler instance
        """
        if scaling_method == "minmax":
            return MinMaxScaler(feature_range=feature_range)
        elif scaling_method == "standard":
            return StandardScaler()
        elif scaling_method == "robust":
            return RobustScaler()
        elif scaling_method == "none":
            return None
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def split_and_scale(
        self,
        df: pd.DataFrame,
        splits: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        scaling_method: str = "minmax",
        feature_range: Tuple[float, float] = (0.0, 1.0),
        columns_to_scale: Optional[list] = None,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Any]]:
        """
        Split data and apply scaling.
        
        Scaler is fit ONLY on training data, then applied to all splits.
        
        Args:
            df: Input DataFrame
            splits: (train, val, test) split ratios
            scaling_method: "minmax", "standard", "robust", or "none"
            feature_range: Range for MinMaxScaler
            columns_to_scale: Columns to scale (None = all numeric columns)
            shuffle: Whether to shuffle before splitting
            random_state: Random seed
            
        Returns:
            (train_df, val_df, test_df, scaler) tuple
        """
        # Split data
        train_df, val_df, test_df = self.split_data(df, splits, shuffle, random_state)
        
        # Determine columns to scale
        if columns_to_scale is None:
            columns_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create and fit scaler on training data
        scaler = self.create_scaler(scaling_method, feature_range)
        
        if scaler is not None:
            # Fit on training data only
            scaler.fit(train_df[columns_to_scale])
            
            # Transform all splits
            train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
            val_df[columns_to_scale] = scaler.transform(val_df[columns_to_scale])
            test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
            
            print(f"✓ Applied {scaling_method} scaling to {len(columns_to_scale)} columns")
        else:
            print(f"✓ No scaling applied")
        
        return train_df, val_df, test_df, scaler
    
    # =========================================================================
    # Data Export
    # =========================================================================
    
    def to_numpy(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert DataFrame to numpy arrays.
        
        Args:
            df: Input DataFrame
            feature_cols: Feature column names (None = all except target)
            target_col: Target column name (None = no target)
            
        Returns:
            (X, y) tuple where y is None if no target_col specified
        """
        if feature_cols is None:
            if target_col:
                feature_cols = [col for col in df.columns if col != target_col]
            else:
                feature_cols = df.columns.tolist()
        
        X = df[feature_cols].values.astype(np.float32)
        
        if target_col:
            y = df[target_col].values.astype(np.float32)
        else:
            y = None
        
        print(f"✓ Converted to numpy:")
        print(f"  X shape: {X.shape}")
        if y is not None:
            print(f"  y shape: {y.shape}")
        
        return X, y
    
    def to_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert DataFrame to dictionary of numpy arrays.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to numpy arrays
        """
        return {col: df[col].values.astype(np.float32) for col in df.columns}
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def validate_data(
        self,
        df: pd.DataFrame,
        required_columns: Optional[list] = None,
        check_nulls: bool = True
    ) -> bool:
        """
        Validate DataFrame for ML readiness.
        
        Args:
            df: DataFrame to validate
            required_columns: Required column names
            check_nulls: Whether to check for null values
            
        Returns:
            True if valid
            
        Raises:
            ValueError if validation fails
        """
        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        # Check for nulls
        if check_nulls:
            null_counts = df.isnull().sum()
            if null_counts.any():
                print("⚠ Warning: Null values detected:")
                for col, count in null_counts[null_counts > 0].items():
                    print(f"  {col}: {count:,} nulls ({count/len(df):.1%})")
        
        print(f"✓ Data validation passed")
        return True

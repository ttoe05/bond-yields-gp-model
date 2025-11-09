"""
Data loading and validation utilities for bond yield forecasting.
"""
from pyexpat import features

import pandas as pd
import numpy as np
import yaml
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BondDataLoader:
    """Handles loading and validation of bond yield data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the bond_train_diff.parquet file
        """
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that the data path exists."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if not self.data_path.suffix == '.parquet':
            raise ValueError("Data file must be a .parquet file")
    
    def load_data(self, x: List[str], y: List[str], actuals: List[str]) -> None:
        """
        Load the bond data from parquet file.
        
        Returns:
            DataFrame with bond yield data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            df_tmp = pd.read_parquet(self.data_path)
            
            # Ensure datetime index
            # if not isinstance(self.data.index, pd.DatetimeIndex):
            #     if 'date' in self.data.columns:
            #         self.data.set_index('date', inplace=True)
            #     else:
            #         logger.warning("No datetime index found, assuming first column is date")
            #         self.data.index = pd.to_datetime(self.data.index)
            #
            # Sort by date
            df_tmp.sort_index(inplace=True)
            full_columns = x + y + actuals
            # handle nulls
            df_tmp = df_tmp[full_columns].dropna()
            logger.info(f"Loaded data: {df_tmp.shape[0]} rows, {df_tmp.shape[1]} columns")
            logger.info(f"Date range: {df_tmp.index.min()} to {df_tmp.index.max()}")
            
            self.data = df_tmp
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError

    
    def get_time_windows(self, window_size: int = 252, min_window_size: int = 100) -> List[Tuple[int, int]]:
        """
        Generate time windows for walk-forward validation.
        
        Args:
            window_size: Size of the training window
            min_window_size: Minimum window size required
            
        Returns:
            List of (start_idx, end_idx) tuples for training windows
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        n_samples = len(self.data)
        windows = []
        
        # Start from minimum window size and expand until we reach full window size
        for i in range(max(window_size, min_window_size), n_samples):
            start_idx = max(0, i - window_size)
            end_idx = i
            windows.append((start_idx, end_idx))
        
        logger.info(f"Generated {len(windows)} time windows")
        return windows
    
    def get_window_data(self, start_idx: int, end_idx: int, 
                       target_columns: list[str], feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract data for a specific time window.
        
        Args:
            start_idx: Start index of the window
            end_idx: End index of the window
            target_column: Name of target variable
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Extract features and target
        x = window_data[feature_columns]
        y = window_data[target_columns]
        
        # # Handle missing values by forward filling
        # X = X.fillna(method='ffill').fillna(method='bfill')
        # y = y.fillna(method='ffill').fillna(method='bfill')
        
        return x, y
    
    def get_prediction_point(self, idx: int, feature_columns: List[str]) -> pd.DataFrame:
        """
        Get features for a single prediction point.
        
        Args:
            idx: Index of the prediction point
            feature_columns: List of feature column names
            
        Returns:
            Series with features for prediction
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for data with {len(self.data)} rows")
        
        features = self.data.iloc[idx][feature_columns]
        
        return features.to_frame().T  # Return as DataFrame
    
    def get_actual_value(self, idx: int, target_columns: list[str]) -> float:
        """
        Get actual target value for a prediction point.
        
        Args:
            idx: Index of the prediction point
            target_columns: Name of target variables
            
        Returns:
            Actual target value
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for data with {len(self.data)} rows")
        
        return self.data.iloc[idx][target_columns].to_frame().T
    
    def get_date_for_index(self, idx: int) -> pd.Timestamp:
        """
        Get the date for a given index.
        
        Args:
            idx: Index position
            
        Returns:
            Date corresponding to the index
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        return self.data.index[idx]
    
    def get_current_yield_value(self, idx: int, base_yield_name: str) -> float:
        """
        Get the current yield value for a specific bond at given index.
        
        Args:
            idx: Index position
            base_yield_name: Base yield column name (e.g., 'DGS1MO')
            
        Returns:
            Current yield value for the base yield
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for data with {len(self.data)} rows")
        
        if base_yield_name not in self.data.columns:
            raise ValueError(f"Base yield column '{base_yield_name}' not found in data")
        
        return self.data.iloc[idx][base_yield_name]
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for the loaded data.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        return self.data.describe()

if __name__ == "__main__":
    # Example usage
    data_loader = BondDataLoader(data_path="/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/bond_yields_ns_params_shifted_1.parquet")
    # print(data_loader.summary_statistics())
    # read in the yaml file to get the feature columns and the target columns
    config = yaml.safe_load(Path("/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/features_selected.yaml").read_text())
    target_vars = config['dependent_variables']
    features = config['one-day-ahead']['max_features']
    data_loader.load_data(x=features, y=target_vars)
    windows = data_loader.get_time_windows(window_size=252, min_window_size=100)
    print(f"First 5 time windows: {windows[:5]}")
    print(f"Target Variables: {target_vars}")
    print(f"Feature Variables: {features}")
    x, y = data_loader.get_window_data(start_idx=0, end_idx=252, target_columns=target_vars, feature_columns=features)
    print(f"Features shape: {x.shape}, Target shape: {y.shape}")
    pred_features = data_loader.get_prediction_point(idx=252, feature_columns=features)
    print(f"Prediction features: {pred_features}")
    actual_value = data_loader.get_actual_value(idx=252, target_columns=target_vars)
    print(f"Actual target value: {actual_value}")
    pred_date = data_loader.get_date_for_index(idx=252)
    print(f"Prediction date: {pred_date}")
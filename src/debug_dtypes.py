"""
Debug data types to find scaling issues.
"""

import logging
import sys
import pandas as pd
import numpy as np
sys.path.append('.')

from data_loader import BondDataLoader
from feature_manager import FeatureManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_dtypes():
    """Debug data types in the dataset."""
    
    config_file = 'data/features_selected4.yaml'
    data_file = 'data/kernel_regression_train_shifted_7_.parquet'
    time_prediction = 'seven-day-ahead'
    
    # Initialize components
    feature_manager = FeatureManager(features_config_path=config_file)
    data_loader = BondDataLoader(data_path=data_file)
    
    # Get features and targets
    features = feature_manager.get_features_for_time_pred(time_prediction=time_prediction)
    target_columns = feature_manager.get_dependent_variables(time_prediction=time_prediction)
    
    # Load data
    data_loader.load_data(x=features, y=target_columns)
    
    logger.info("Data dtypes:")
    for col in data_loader.data.columns:
        dtype = data_loader.data[col].dtype
        logger.info(f"  {col}: {dtype}")
        
        # Check for problematic values
        if dtype == 'object':
            logger.warning(f"    Column {col} has object dtype!")
            unique_vals = data_loader.data[col].unique()[:10]
            logger.info(f"    Sample values: {unique_vals}")
        elif dtype in ['int64', 'float64']:
            logger.info(f"    Column {col} is numeric ✓")
        else:
            logger.warning(f"    Column {col} has unusual dtype: {dtype}")
    
    # Test with first window
    windows = data_loader.get_time_windows(window_size=100, min_window_size=50)
    start_idx, end_idx = windows[0]
    
    x_train, y_train = data_loader.get_window_data(
        start_idx=start_idx,
        end_idx=end_idx,
        target_columns=target_columns,
        feature_columns=features
    )
    
    logger.info("X_train dtypes:")
    for col in x_train.columns:
        dtype = x_train[col].dtype
        logger.info(f"  {col}: {dtype}")
        if dtype == 'object':
            logger.error(f"    ERROR: X_train column {col} is object type!")
    
    logger.info("Y_train dtypes:")
    for col in y_train.columns:
        dtype = y_train[col].dtype
        logger.info(f"  {col}: {dtype}")
        if dtype == 'object':
            logger.error(f"    ERROR: Y_train column {col} is object type!")
    
    # Try manual scaling
    from sklearn.preprocessing import StandardScaler
    
    logger.info("Testing manual scaling...")
    try:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_train)
        logger.info("✅ Y scaling works manually")
    except Exception as e:
        logger.error(f"❌ Y scaling fails manually: {str(e)}")
        
        # Check for infinities, NaN, etc.
        logger.info("Checking for problematic values in y_train:")
        logger.info(f"  NaN values: {y_train.isna().sum().sum()}")
        logger.info(f"  Infinite values: {np.isinf(y_train.select_dtypes(include=[np.number])).sum().sum()}")
        logger.info(f"  Empty columns: {(y_train.shape[0] == 0)}")
        
        # Check each column individually
        for col in y_train.columns:
            try:
                scaler_col = StandardScaler()
                col_data = y_train[[col]]
                logger.info(f"  Testing column {col}...")
                scaler_col.fit_transform(col_data)
                logger.info(f"    ✅ Column {col} scales OK")
            except Exception as col_e:
                logger.error(f"    ❌ Column {col} fails: {str(col_e)}")
                logger.info(f"    Column dtype: {y_train[col].dtype}")
                logger.info(f"    Column values: {y_train[col].head()}")

if __name__ == "__main__":
    debug_dtypes()
"""
Debug window data extraction to see why y_train is empty.
"""

import logging
import sys
sys.path.append('.')

from data_loader import BondDataLoader
from feature_manager import FeatureManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_window_data():
    """Debug window data extraction."""
    
    config_file = 'data/features_selected4.yaml'
    data_file = 'data/kernel_regression_train_shifted_7_.parquet'
    time_prediction = 'seven-day-ahead'
    
    # Initialize components
    feature_manager = FeatureManager(features_config_path=config_file)
    data_loader = BondDataLoader(data_path=data_file)
    
    # Get features and targets
    features = feature_manager.get_features_for_time_pred(time_prediction=time_prediction)
    target_columns = feature_manager.get_dependent_variables(time_prediction=time_prediction)
    
    logger.info(f"Features: {features}")
    logger.info(f"Targets: {target_columns}")
    
    # Load data
    data_loader.load_data(x=features, y=target_columns)
    logger.info(f"Data shape: {data_loader.data.shape}")
    logger.info(f"Data columns: {list(data_loader.data.columns)}")
    
    # Check if target columns exist in the loaded data
    missing_targets = [col for col in target_columns if col not in data_loader.data.columns]
    if missing_targets:
        logger.error(f"Missing target columns in data: {missing_targets}")
    else:
        logger.info("✅ All target columns found in data")
    
    # Get first few windows to debug
    windows = data_loader.get_time_windows(window_size=100, min_window_size=50)
    logger.info(f"First 3 windows: {windows[:3]}")
    
    # Test extracting data from the first window
    start_idx, end_idx = windows[0]
    logger.info(f"Testing window: {start_idx} to {end_idx}")
    
    try:
        x_train, y_train = data_loader.get_window_data(
            start_idx=start_idx,
            end_idx=end_idx,
            target_columns=target_columns,
            feature_columns=features
        )
        
        logger.info(f"X_train shape: {x_train.shape}")
        logger.info(f"Y_train shape: {y_train.shape}")
        logger.info(f"X_train columns: {list(x_train.columns)}")
        logger.info(f"Y_train columns: {list(y_train.columns)}")
        
        if y_train.empty:
            logger.error("❌ Y_train is empty!")
            logger.info("Checking data at window indices...")
            window_data = data_loader.data.iloc[start_idx:end_idx]
            logger.info(f"Window data shape: {window_data.shape}")
            logger.info(f"Window data columns: {list(window_data.columns)}")
            logger.info("Available target columns in window:")
            for col in target_columns:
                if col in window_data.columns:
                    logger.info(f"  {col}: {window_data[col].isna().sum()} NaN values")
                else:
                    logger.error(f"  {col}: NOT FOUND")
        else:
            logger.info("✅ Y_train has data")
            
    except Exception as e:
        logger.error(f"Error extracting window data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_window_data()
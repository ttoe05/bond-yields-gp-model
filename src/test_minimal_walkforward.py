"""
Minimal test to debug the walk-forward validation issue.
"""

import logging
import sys
import numpy as np
sys.path.append('.')

from data_loader import BondDataLoader
from feature_manager import FeatureManager
from gp_models import GaussianProcessEnsemble
from walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_walkforward_minimal():
    """Test walk-forward with minimal configuration."""
    
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
    
    # Create model
    model = GaussianProcessEnsemble(selection_metric='train_cosine_distance', n_jobs=1)
    
    # Test with minimal walk-forward - just a few windows
    validator = WalkForwardValidator(
        model=model,
        data_loader=data_loader,
        feature_manager=feature_manager,
        time_prediction=time_prediction,
        window_size=100,  # Very small for debugging
        min_window_size=50,
        model_retrain_interval=50,  # Don't retrain often
        window_type='sliding',
        use_scaling=True,
        persist_samples=False,  # Don't persist
        step_size=10,  # Larger step for fewer windows
        n_parallel_jobs=1
    )
    
    # Get windows to see what we're working with
    windows = data_loader.get_time_windows(
        window_size=100, 
        min_window_size=50,
        window_type='sliding'
    )
    logger.info(f"Generated {len(windows)} windows")
    logger.info(f"First 5 windows: {windows[:5]}")
    
    # Test getting window data for the first window
    start_idx, end_idx = windows[0]
    logger.info(f"Testing window: {start_idx} to {end_idx}")
    
    try:
        x_train, y_train = data_loader.get_window_data(
            start_idx=start_idx,
            end_idx=end_idx,
            target_columns=target_columns,
            feature_columns=features
        )
        logger.info(f"Training data shapes: X={x_train.shape}, y={y_train.shape}")
        logger.info(f"X columns: {list(x_train.columns)}")
        logger.info(f"y columns: {list(y_train.columns)}")
        
        # Check for NaN values
        logger.info(f"X has NaN: {x_train.isna().any().any()}")
        logger.info(f"y has NaN: {y_train.isna().any().any()}")
        
        # Test prediction point
        pred_x = data_loader.get_prediction_point(
            idx=end_idx,
            feature_columns=features
        )
        logger.info(f"Prediction point shape: {pred_x.shape}")
        
        logger.info("✅ Window data extraction successful!")
        
    except Exception as e:
        logger.error(f"❌ Window data extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_walkforward_minimal()
    if success:
        print("✅ Minimal walk-forward test passed - ready for full validation")
    else:
        print("❌ Fix window issues first")
        sys.exit(1)
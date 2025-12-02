"""
Debug the walk-forward validation error.
"""

import logging
import sys
import traceback
sys.path.append('.')

from data_loader import BondDataLoader
from feature_manager import FeatureManager
from gp_models import GaussianProcessEnsemble
from walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_walkforward_error():
    """Debug the 'at least one array or dtype is required' error."""
    
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
    
    # Create model
    model = GaussianProcessEnsemble(selection_metric='train_cosine_distance', n_jobs=1)
    
    # Very minimal walk-forward validator to isolate the error
    validator = WalkForwardValidator(
        model=model,
        data_loader=data_loader,
        feature_manager=feature_manager,
        time_prediction=time_prediction,
        window_size=100,  # Very small
        min_window_size=50,
        model_retrain_interval=100,  # No retraining
        window_type='sliding',
        use_scaling=True,  # This might be causing the issue
        persist_samples=False,
        step_size=50,  # Large step for only a few predictions
        n_parallel_jobs=1
    )
    
    try:
        logger.info("Starting minimal walk-forward validation...")
        validator.run_walk_forward_validation()
        logger.info("✅ Walk-forward validation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Walk-forward validation failed: {str(e)}")
        logger.error("Full traceback:")
        traceback.print_exc()
        
        # Try without scaling
        logger.info("Trying without scaling...")
        try:
            validator_no_scaling = WalkForwardValidator(
                model=model,
                data_loader=data_loader,
                feature_manager=feature_manager,
                time_prediction=time_prediction,
                window_size=100,
                min_window_size=50,
                model_retrain_interval=100,
                window_type='sliding',
                use_scaling=False,  # Try without scaling
                persist_samples=False,
                step_size=50,
                n_parallel_jobs=1
            )
            validator_no_scaling.run_walk_forward_validation()
            logger.info("✅ Walk-forward validation WITHOUT SCALING worked!")
            return "scaling_issue"
            
        except Exception as e2:
            logger.error(f"❌ Walk-forward validation failed even without scaling: {str(e2)}")
            traceback.print_exc()
            return "other_issue"

if __name__ == "__main__":
    result = debug_walkforward_error()
    print(f"Debug result: {result}")
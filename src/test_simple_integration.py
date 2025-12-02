"""
Simple integration test to verify imports and basic functionality.
"""

import logging
import sys
sys.path.append('.')

from data_loader import BondDataLoader
from feature_manager import FeatureManager
from gp_models import GaussianProcessEnsemble
from walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_integration():
    """Test basic functionality with minimal configuration."""
    
    try:
        # Test imports and basic initialization
        logger.info("Testing imports...")
        
        config_file = 'data/features_selected4.yaml'
        data_file = 'data/bond_yields_train_shifted_7.parquet'
        time_prediction = 'seven-day-ahead'
        
        # Initialize components
        feature_manager = FeatureManager(features_config_path=config_file)
        data_loader = BondDataLoader(data_path=data_file)
        
        # Get features
        features = feature_manager.get_features_for_time_pred(time_prediction=time_prediction)
        target_columns = feature_manager.get_dependent_variables(time_prediction=time_prediction)
        
        logger.info(f"Features: {features}")
        logger.info(f"Targets: {target_columns}")
        
        # Load data
        data_loader.load_data(x=features, y=target_columns)
        logger.info(f"Loaded data shape: {data_loader.data.shape}")
        
        # Create model
        model = GaussianProcessEnsemble(selection_metric='train_cosine_distance', n_jobs=1)
        logger.info(f"Created model: {type(model)}")
        
        # Test minimal walk-forward setup (don't run full validation)
        validator = WalkForwardValidator(
            model=model,
            data_loader=data_loader,
            feature_manager=feature_manager,
            time_prediction=time_prediction,
            window_size=100,  # Small for testing
            min_window_size=50,
            model_retrain_interval=10,
            window_type='sliding',
            use_scaling=True,
            persist_samples=False,  # Don't persist for this test
            step_size=1,
            n_parallel_jobs=1
        )
        
        logger.info("✅ All components initialized successfully!")
        logger.info("✅ Simple integration test passed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_simple_integration()
    if success:
        print("✅ Ready to run full integration tests")
    else:
        print("❌ Fix issues before running full tests")
        sys.exit(1)
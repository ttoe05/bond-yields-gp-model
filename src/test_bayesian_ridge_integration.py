"""
Test script to verify Bayesian Ridge integration with existing pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from data_loader import BondDataLoader
from feature_manager import FeatureManager
from bayesian_ridge_models import BayesianRidgeEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bayesian_ridge_integration():
    """Test the Bayesian Ridge model integration."""
    
    # Load data (using same paths as gp_models.py)
    data_path = "/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/bond_yields_ns_params_shifted_1.parquet"
    features_config_path = "/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/features_selected.yaml"
    
    try:
        # Initialize data loader and feature manager
        data_loader = BondDataLoader(data_path=data_path)
        feature_manager = FeatureManager(features_config_path=features_config_path)
        
        # Get features and targets
        features = feature_manager.features_config['one-day-ahead']['max_features']
        target_var = feature_manager.get_dependent_variables()
        
        # Load data
        data_loader.load_data(x=features, y=target_var)
        logger.info(f"Loaded dataset with {len(data_loader.data)} rows")
        
        # Get time windows
        windows = data_loader.get_time_windows(window_size=252, min_window_size=100)
        
        # Get training data (using same approach as gp_models.py)
        x_train, y_train = data_loader.get_window_data(
            start_idx=windows[-2][0], 
            end_idx=windows[-2][1],
            target_columns=target_var, 
            feature_columns=features
        )
        logger.info(f"Training data shape: X={x_train.shape}, y={y_train.shape}")
        
        # Get prediction point
        pred_features = data_loader.get_prediction_point(
            idx=windows[-2][1], 
            feature_columns=features
        )
        actual_value = data_loader.get_actual_value(
            idx=windows[-2][1], 
            target_columns=target_var
        )
        
        # Create Bayesian Ridge ensemble
        br_ensemble = BayesianRidgeEnsemble(selection_metric='train_cosine_distance')
        
        # Train models
        logger.info("Training Bayesian Ridge models...")
        br_ensemble.train_models(x=x_train, y=y_train)
        
        # Make predictions
        y_pred, std_val = br_ensemble.predict_val(x=pred_features, return_std=True)
        y_samples = br_ensemble.predict_val_distribution(x=pred_features, n_samples=1000)
        
        logger.info(f"Prediction: {y_pred}")
        logger.info(f"Actual: {actual_value.to_numpy()}")
        logger.info(f"Y samples shape: {y_samples.shape}")
        
        # Get model summary
        model_summary = br_ensemble.get_model_summary()
        logger.info(f"Best alpha: {model_summary['best_alpha']}")
        
        # Get feature importance
        feature_importance = br_ensemble.get_feature_importance_proxy(X=x_train)
        logger.info(f"Top 5 features:\n{feature_importance.sort_values(ascending=False).head()}")
        
        logger.info("✅ Bayesian Ridge integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_bayesian_ridge_integration()
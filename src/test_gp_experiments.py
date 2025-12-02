"""
Test script for GPModelingExperiments class.
"""

import pandas as pd
import numpy as np
from gp_experiments import GPModelingExperiments
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality of GPModelingExperiments class."""
    logger.info("Starting basic functionality test")
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    n_targets = 3
    
    # Generate synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate synthetic targets with some correlation to features
    y_data = np.random.randn(n_samples, n_targets) + 0.5 * X.sum(axis=1).values.reshape(-1, 1)
    y = pd.Series(y_data.flatten()[:n_samples])  # Single target for simplicity
    
    logger.info(f"Created synthetic data: X{X.shape}, y{y.shape}")
    
    # Initialize experiments
    gp_experiments = GPModelingExperiments(
        selection_metric='train_r2_avg',
        random_state=42,
        normalize_features=True
    )
    
    logger.info(f"Initialized GPModelingExperiments with {len(gp_experiments.kernels)} kernels")
    
    # Test individual methods
    logger.info("Testing transform_features method...")
    scaler, x_transformed = gp_experiments.transform_features(X)
    logger.info(f"Feature transformation successful. Shape: {x_transformed.shape}")
    
    # Test train_gp_model with single kernel
    logger.info("Testing train_gp_model method...")
    kernel = gp_experiments.kernels['RBF']
    training_result = gp_experiments.train_gp_model(X, y, kernel)
    logger.info(f"Training successful. Metrics: {list(training_result['training_metrics'].keys())}")
    
    # Test predict method
    logger.info("Testing predict method...")
    # Use a subset for prediction
    X_pred = X.iloc[:10]
    prediction_result = gp_experiments.predict(X_pred)
    logger.info(f"Prediction successful. Shape: {prediction_result['predictions'].shape}")
    
    # Test predict_samples method
    logger.info("Testing predict_samples method...")
    samples = gp_experiments.predict_samples(X_pred, n_samples=100)
    logger.info(f"Sampling successful. Shape: {samples.shape}")
    
    # Test fit_transform_predict_pipeline
    logger.info("Testing fit_transform_predict_pipeline method...")
    training_metrics, predict_metrics, y_samples = gp_experiments.fit_transform_predict_pipeline(
        x_train=X,
        y_train=y,
        kernel=kernel,
        x_predict=X_pred,
        samples=True
    )
    logger.info("Pipeline test successful")
    
    logger.info("Basic functionality test completed successfully!")
    return True

def test_kernel_experiments():
    """Test kernel experiments functionality."""
    logger.info("Starting kernel experiments test")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 50  # Smaller for faster testing
    n_features = 3
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y = pd.Series(np.random.randn(n_samples))
    
    # Initialize experiments with limited kernels for faster testing
    gp_experiments = GPModelingExperiments(
        selection_metric='train_r2_avg',
        random_state=42
    )
    
    # Override with fewer kernels for testing
    gp_experiments.kernels = {
        'RBF': gp_experiments.kernels['RBF'],
        'RationalQuadratic': gp_experiments.kernels['RationalQuadratic']
    }
    
    logger.info(f"Testing with {len(gp_experiments.kernels)} kernels")
    
    # Run kernel experiments
    results = gp_experiments.run_kernel_experiments(X, y)
    logger.info(f"Kernel experiments completed. Results for {len(results)} kernels")
    
    # Test model selection
    best_kernel = gp_experiments.select_best_model()
    logger.info(f"Best kernel selected: {best_kernel}")
    
    # Test rankings
    rankings = gp_experiments.get_kernel_rankings()
    logger.info(f"Rankings table shape: {rankings.shape}")
    print(rankings)
    
    # Test model summary
    summary = gp_experiments.get_model_summary()
    logger.info(f"Model summary keys: {list(summary.keys())}")
    
    logger.info("Kernel experiments test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        # Run basic tests
        test_basic_functionality()
        
        # Run kernel experiments test
        test_kernel_experiments()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise
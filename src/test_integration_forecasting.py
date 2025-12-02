"""
Integration test script for main_forecasting.py with all three models:
- GP model: rolling window, size 2400, scaling enabled
- Bayesian Ridge: growing window, size 1500, scaling enabled
- Kernel Ridge: growing window, size 1500, scaling enabled

Tests seven-day-ahead and thirty-day-ahead predictions.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.append('.')

from data_loader import BondDataLoader
from feature_manager import FeatureManager
from gp_models import GaussianProcessEnsemble
from bayesian_ridge_models import BayesianRidgeEnsemble
from kernel_ridge_models import KernelRidgeEnsemble
from walk_forward import WalkForwardValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IntegrationTestPipeline:
    """Integration test pipeline for all three models."""
    
    def __init__(self, model_name: str, time_prediction: str, config_file: str, data_file: str,
                 window_size: int, min_window_size: int, retrain_interval: int,
                 window_type: str = 'sliding', use_scaling: bool = True,
                 selection_metric: str = 'train_cosine_distance', n_jobs: int = 3) -> None:
        """
        Initialize the test pipeline.
        
        Args:
            model_name: 'GP', 'BayesianRidge', or 'KernelRidge'
            time_prediction: Time prediction horizon
            config_file: Path to features configuration
            data_file: Path to data file
            window_size: Training window size
            min_window_size: Minimum training window size
            retrain_interval: Model retrain interval
            window_type: 'sliding' or 'growing'
            use_scaling: Whether to use scaling
            selection_metric: Model selection metric
            n_jobs: Number of parallel jobs
        """
        self.model_name = model_name
        self.time_prediction = time_prediction
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.retrain_interval = retrain_interval
        self.window_type = window_type
        self.use_scaling = use_scaling
        
        # Initialize components
        self.feature_manager = FeatureManager(features_config_path=config_file)
        self.data_loader = BondDataLoader(data_path=data_file)
        
        # Create model instance
        if model_name == 'GP':
            self.model_obj = GaussianProcessEnsemble(
                selection_metric=selection_metric, 
                n_jobs=n_jobs
            )
        elif model_name == 'BayesianRidge':
            self.model_obj = BayesianRidgeEnsemble(
                selection_metric=selection_metric,
                n_jobs=n_jobs
            )
        elif model_name == 'KernelRidge':
            self.model_obj = KernelRidgeEnsemble(
                training_metric='mse',  # Using MSE for KRR
                random_state=42,
                n_jobs=n_jobs
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Must be 'GP', 'BayesianRidge', or 'KernelRidge'")

        logger.info(
            f"Initialized {model_name} test pipeline for {time_prediction} "
            f"with window_size={window_size}, window_type={window_type}, "
            f"scaling={use_scaling}"
        )
        
    def run_test(self) -> None:
        """Run the integration test."""
        logger.info(f"Starting {self.model_name} integration test for {self.time_prediction}")
        
        try:
            # Get features for the specified time prediction
            features = self.feature_manager.get_features_for_time_pred(
                time_prediction=self.time_prediction
            )
            target_columns = self.feature_manager.get_dependent_variables(
                time_prediction=self.time_prediction
            )
            
            # Load data
            self.data_loader.load_data(x=features, y=target_columns)
            logger.info(f"Loaded data with {len(self.data_loader.data)} rows")
            logger.info(f"Using {len(features)} features: {features}")
            logger.info(f"Predicting {len(target_columns)} targets: {target_columns}")

            # Set up walk-forward validator
            wf_validator = WalkForwardValidator(
                model=self.model_obj,
                data_loader=self.data_loader,
                feature_manager=self.feature_manager,
                time_prediction=self.time_prediction,
                window_size=self.window_size,
                min_window_size=self.min_window_size,
                model_retrain_interval=self.retrain_interval,
                window_type=self.window_type,
                use_scaling=self.use_scaling,
                persist_samples=True,
                step_size=1,
                n_parallel_jobs=1  # Single job for testing
            )

            # Execute walk-forward validation
            logger.info("Starting walk-forward validation...")
            wf_validator.run_walk_forward_validation()
            
            # Export results
            output_dir = f'results/{self.time_prediction}_{self.model_name.lower()}/'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            wf_validator.export_results(filepath=output_dir)
            
            logger.info(f"✅ {self.model_name} test completed successfully for {self.time_prediction}")
            logger.info(f"Results exported to: {output_dir}")
            
            # Verify outputs exist
            self._verify_outputs(output_dir)
            
        except Exception as e:
            logger.error(f"❌ {self.model_name} test failed for {self.time_prediction}: {str(e)}")
            raise

    def _verify_outputs(self, output_dir: str) -> None:
        """Verify that expected output files were created."""
        output_path = Path(output_dir)
        
        expected_files = [
            f"{self.time_prediction}_{self.model_name.lower()}_dgs_results.parquet",
            f"{self.time_prediction}_{self.model_name.lower()}_mean_yields.parquet", 
            f"{self.time_prediction}_{self.model_name.lower()}_model_summary.parquet"
        ]
        
        samples_dir = Path(f"results/{self.time_prediction}/samples/dgs_yields")
        
        logger.info("Verifying output files...")
        
        for expected_file in expected_files:
            file_path = output_path / expected_file
            if file_path.exists():
                logger.info(f"✅ Found: {expected_file}")
            else:
                logger.warning(f"⚠️ Missing: {expected_file}")
        
        # Check if samples directory exists and has files
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.parquet"))
            if sample_files:
                logger.info(f"✅ Found {len(sample_files)} sample files in {samples_dir}")
            else:
                logger.warning(f"⚠️ No sample files found in {samples_dir}")
        else:
            logger.warning(f"⚠️ Samples directory not found: {samples_dir}")


def run_integration_tests():
    """Run all integration tests."""
    start_time = time.time()
    
    # Test configurations - using small windows for fast testing
    test_configs = [
        {
            'model_name': 'GP',
            'window_size': 200,  # Small window for testing
            'min_window_size': 100,
            'window_type': 'sliding',  # Rolling = sliding
            'retrain_interval': 20  # Less frequent for integration test
        },
        {
            'model_name': 'BayesianRidge', 
            'window_size': 150,  # Small window for testing
            'min_window_size': 75,
            'window_type': 'growing',  # Growing window as requested
            'retrain_interval': 15
        },
        {
            'model_name': 'KernelRidge',
            'window_size': 150,  # Small window for testing
            'min_window_size': 75,
            'window_type': 'growing',  # Growing window as requested
            'retrain_interval': 15
        }
    ]
    
    # Time predictions to test
    time_predictions = ['seven-day-ahead', 'thirty-day-ahead']
    data_files = {
        'seven-day-ahead': 'data/kernel_regression_train_shifted_7_.parquet',
        'thirty-day-ahead': 'data/kernel_regression_train_shifted_30_.parquet'
    }
    
    config_file = 'data/features_selected3.yaml'
    
    total_tests = len(test_configs) * len(time_predictions)
    completed_tests = 0
    failed_tests = []
    
    logger.info(f"Starting integration tests: {total_tests} total tests")
    logger.info("=" * 60)
    
    for time_pred in time_predictions:
        data_file = data_files[time_pred]
        
        logger.info(f"\n🔄 Testing {time_pred} predictions using {data_file}")
        logger.info("-" * 40)
        
        for config in test_configs:
            test_name = f"{config['model_name']}_{time_pred}"
            
            try:
                pipeline = IntegrationTestPipeline(
                    model_name=config['model_name'],
                    time_prediction=time_pred,
                    config_file=config_file,
                    data_file=data_file,
                    window_size=config['window_size'],
                    min_window_size=config['min_window_size'],
                    retrain_interval=config['retrain_interval'],
                    window_type=config['window_type'],
                    use_scaling=True,
                    selection_metric='train_cosine_distance',
                    n_jobs=1
                )
                
                pipeline.run_test()
                completed_tests += 1
                
            except Exception as e:
                failed_tests.append(test_name)
                logger.error(f"❌ Test {test_name} failed: {str(e)}")
                continue
    
    # Summary
    end_time = time.time()
    duration = (end_time - start_time) / 60  # minutes
    
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Completed: {completed_tests}")
    logger.info(f"Failed: {len(failed_tests)}")
    logger.info(f"Duration: {duration:.2f} minutes")
    
    if failed_tests:
        logger.error(f"Failed tests: {', '.join(failed_tests)}")
    else:
        logger.info("🎉 All tests passed!")


if __name__ == "__main__":
    run_integration_tests()
"""
Walk-forward validation framework for bond yield forecasting.
Implements sliding window validation with proper time series handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from datetime import datetime
import gc
from .data_loader import BondDataLoader
from .feature_manager import FeatureManager
from .gp_models import GaussianProcessEnsemble
from .parallel_trainer import ParallelGPTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation framework for time series forecasting."""
    
    def __init__(self, 
                 data_loader: BondDataLoader,
                 feature_manager: FeatureManager,
                 window_size: int = 252,
                 min_window_size: int = 60,
                 step_size: int = 1,
                 model_retrain_interval: int = 20,
                 n_parallel_jobs: int = 4):
        """
        Initialize walk-forward validator.
        
        Args:
            data_loader: Configured data loader
            feature_manager: Feature manager instance
            window_size: Size of training window (default: 252 trading days)
            min_window_size: Minimum window size for training
            step_size: Step size for walking forward (default: 1 day)
            n_parallel_jobs: Number of parallel jobs for training
        """
        self.data_loader = data_loader
        self.feature_manager = feature_manager
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.step_size = step_size
        self.n_parallel_jobs = n_parallel_jobs
        self.model_retrain_interval = model_retrain_interval
        self.model_retrain_counter = 0
        
        # Results storage
        self.results: Dict[str, List] = {
            'dates': [],
            'actual_values': [],
            'predictions': [],
            'prediction_std': [],
            'best_kernels': [],
            'training_windows': [],
            'feature_counts': [],
            'kernel_evaluation_results': [],
            'training_time': []
        }
        
        # Model tracking
        self.model_summaries: List[Dict] = []
        
        logger.info(f"Initialized walk-forward validator: window_size={window_size}, "
                   f"min_window_size={min_window_size}, step_size={step_size}")
    
    def validate_setup(self, bond_index: str) -> Dict[str, Any]:
        """
        Validate that setup is ready for walk-forward validation.
        
        Args:
            bond_index: Bond index to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'bond_index': bond_index,
            'data_loaded': self.data_loader.data is not None,
            'features_available': False,
            'sufficient_data': False,
            'validation_passed': False,
            'data_shape': None,
            'feature_count': 0,
            'available_windows': 0
        }
        
        try:
            # Check data
            if self.data_loader.data is None:
                logger.error("Data not loaded in data_loader")
                return validation_results
            
            validation_results['data_shape'] = self.data_loader.data.shape
            
            # Check features
            features = self.feature_manager.get_features_for_bond(bond_index)
            available_features = self.feature_manager.filter_features_by_availability(
                bond_index, self.data_loader.data.columns.tolist())
            
            validation_results['features_available'] = len(available_features) > 0
            validation_results['feature_count'] = len(available_features)
            
            # Check sufficient data
            windows = self.data_loader.get_time_windows(self.window_size, self.min_window_size)
            validation_results['available_windows'] = len(windows)
            validation_results['sufficient_data'] = len(windows) > 0
            
            # Overall validation
            validation_results['validation_passed'] = all([
                validation_results['data_loaded'],
                validation_results['features_available'],
                validation_results['sufficient_data']
            ])
            
            if validation_results['validation_passed']:
                logger.info(f"Validation passed for {bond_index}: "
                           f"{validation_results['feature_count']} features, "
                           f"{validation_results['available_windows']} windows")
            else:
                logger.warning(f"Validation failed for {bond_index}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in validation setup: {str(e)}")
            validation_results['error'] = str(e)
            return validation_results
    
    def run_single_prediction(self, 
                            train_start_idx: int,
                            train_end_idx: int, 
                            predict_idx: int,
                            bond_index: str,
                            features: List[str]) -> Dict[str, Any]:
        """
        Run a single prediction step in walk-forward validation.
        
        Args:
            train_start_idx: Start index of training window
            train_end_idx: End index of training window  
            predict_idx: Index to make prediction for
            bond_index: Bond index name
            features: List of feature names
            
        Returns:
            Dictionary with prediction results
        """
        start_time = datetime.now()
        
        try:
            # Get training data
            X_train, y_train = self.data_loader.get_window_data(
                train_start_idx, train_end_idx, bond_index, features)
            
            # Get prediction features
            X_predict = self.data_loader.get_prediction_point(predict_idx, features)
            X_predict_df = pd.DataFrame([X_predict], columns=features)
            
            # Get actual value
            actual_value = self.data_loader.get_actual_value(predict_idx, bond_index)
            
            # Get prediction date
            predict_date = self.data_loader.get_date_for_index(predict_idx)
            
            # Train models in parallel and select best
            parallel_trainer = ParallelGPTrainer(n_jobs=self.n_parallel_jobs)
            
            best_kernel, kernel_results = parallel_trainer.train_and_select_best_kernel(
                X_train, y_train)
            
            # Create final model with best kernel
            gp_ensemble = GaussianProcessEnsemble()
            final_model = gp_ensemble.create_gp_model(best_kernel)
            final_model.fit(X_train, y_train)
            
            # Make prediction with uncertainty
            prediction, prediction_std = final_model.predict(X_predict_df, return_std=True)
            
            # Clean up memory
            del X_train, y_train, parallel_trainer
            gc.collect()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'date': predict_date,
                'actual_value': actual_value,
                'prediction': prediction[0],
                'prediction_std': prediction_std[0],
                'best_kernel': best_kernel,
                'training_window_size': train_end_idx - train_start_idx,
                'feature_count': len(features),
                'kernel_results': kernel_results,
                'training_time': training_time,
                'train_start_date': self.data_loader.get_date_for_index(train_start_idx),
                'train_end_date': self.data_loader.get_date_for_index(train_end_idx - 1)
            }
            
            logger.debug(f"Prediction completed for {predict_date}: "
                        f"actual={actual_value:.4f}, pred={prediction[0]:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single prediction: {str(e)}")
            return {
                'date': None,
                'actual_value': np.nan,
                'prediction': np.nan,
                'prediction_std': np.nan,
                'best_kernel': 'failed',
                'training_window_size': 0,
                'feature_count': len(features),
                'kernel_results': {},
                'training_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e)
            }
    
    def run_walk_forward_validation(self, 
                                  bond_index: str,
                                  max_predictions: Optional[int] = None,
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run complete walk-forward validation.
        
        Args:
            bond_index: Bond index to forecast
            max_predictions: Maximum number of predictions (for testing)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary with complete validation results
        """
        logger.info(f"Starting walk-forward validation for {bond_index}")
        
        # Validate setup
        validation = self.validate_setup(bond_index)
        if not validation['validation_passed']:
            logger.error(f"Setup validation failed: {validation}")
            return validation
        
        # Get features
        features = self.feature_manager.filter_features_by_availability(
            bond_index, self.data_loader.data.columns.tolist())
        
        # Get time windows
        windows = self.data_loader.get_time_windows(self.window_size, self.min_window_size)
        
        if max_predictions:
            windows = windows[:max_predictions]
        
        logger.info(f"Running {len(windows)} predictions with {len(features)} features")
        
        # Initialize results
        self.results = {
            'dates': [],
            'actual_values': [],
            'predictions': [],
            'prediction_std': [],
            'best_kernels': [],
            'training_windows': [],
            'feature_counts': [],
            'kernel_evaluation_results': [],
            'training_time': []
        }
        
        total_predictions = len(windows)
        successful_predictions = 0
        
        # Run predictions
        for i, (train_start_idx, train_end_idx) in enumerate(windows):
            predict_idx = train_end_idx
            
            # Safety check for prediction index
            if predict_idx >= len(self.data_loader.data):
                logger.warning(f"Prediction index {predict_idx} out of bounds, stopping")
                break
            
            logger.info(f"Prediction {i+1}/{total_predictions}: "
                       f"training window [{train_start_idx}:{train_end_idx}], "
                       f"predicting {predict_idx}")
            
            # Run single prediction
            result = self.run_single_prediction(
                train_start_idx, train_end_idx, predict_idx, bond_index, features)
            
            # Store results
            if not pd.isna(result['prediction']):
                self.results['dates'].append(result['date'])
                self.results['actual_values'].append(result['actual_value'])
                self.results['predictions'].append(result['prediction'])
                self.results['prediction_std'].append(result['prediction_std'])
                self.results['best_kernels'].append(result['best_kernel'])
                self.results['training_windows'].append(result['training_window_size'])
                self.results['feature_counts'].append(result['feature_count'])
                self.results['kernel_evaluation_results'].append(result['kernel_results'])
                self.results['training_time'].append(result['training_time'])
                
                successful_predictions += 1
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_predictions, result)
            
            # Memory management
            if (i + 1) % 10 == 0:
                gc.collect()
        
        # Create summary
        summary = self._create_validation_summary(bond_index, successful_predictions, total_predictions)
        
        logger.info(f"Walk-forward validation completed: {successful_predictions}/{total_predictions} successful predictions")
        
        return {
            'validation_results': validation,
            'prediction_results': self.results,
            'summary': summary,
            'bond_index': bond_index,
            'features_used': features,
            'parameters': {
                'window_size': self.window_size,
                'min_window_size': self.min_window_size,
                'step_size': self.step_size,
                'n_parallel_jobs': self.n_parallel_jobs
            }
        }
    
    def _create_validation_summary(self, bond_index: str, 
                                 successful_predictions: int, 
                                 total_predictions: int) -> Dict[str, Any]:
        """Create summary of validation results."""
        
        if len(self.results['predictions']) == 0:
            return {'error': 'No successful predictions'}
        
        predictions = np.array(self.results['predictions'])
        actuals = np.array(self.results['actual_values'])
        std_devs = np.array(self.results['prediction_std'])
        
        # Calculate basic metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        directional_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0
        
        # Kernel selection frequency
        kernel_counts = pd.Series(self.results['best_kernels']).value_counts().to_dict()
        
        summary = {
            'bond_index': bond_index,
            'successful_predictions': successful_predictions,
            'total_predictions': total_predictions,
            'success_rate': successful_predictions / total_predictions if total_predictions > 0 else 0,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_prediction_std': np.mean(std_devs),
            'directional_accuracy': directional_accuracy,
            'kernel_selection_frequency': kernel_counts,
            'average_training_time': np.mean(self.results['training_time']),
            'total_training_time': np.sum(self.results['training_time']),
            'date_range': (min(self.results['dates']), max(self.results['dates'])) if self.results['dates'] else None
        }
        
        return summary
    
    def export_results(self, filepath: str) -> None:
        """
        Export validation results to file.
        
        Args:
            filepath: Path to save results
        """
        try:
            # Create results DataFrame
            results_df = pd.DataFrame({
                'date': self.results['dates'],
                'actual': self.results['actual_values'],
                'prediction': self.results['predictions'],
                'prediction_std': self.results['prediction_std'],
                'best_kernel': self.results['best_kernels'],
                'training_window_size': self.results['training_windows'],
                'training_time': self.results['training_time']
            })
            
            results_df.to_csv(filepath, index=False)
            logger.info(f"Results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise
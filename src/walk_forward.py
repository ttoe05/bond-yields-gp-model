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
from data_loader import BondDataLoader
from feature_manager import FeatureManager
from gp_models import GaussianProcessEnsemble
from bayesian_ridge_models import BayesianRidgeEnsemble
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation framework for time series forecasting."""
    
    def __init__(self,
                 model: GaussianProcessEnsemble | BayesianRidgeEnsemble,
                 data_loader: BondDataLoader,
                 feature_manager: FeatureManager,
                 time_prediction: str,
                 persist_samples: bool = True,
                 window_size: int = 3000,
                 min_window_size: int = 2000,
                 step_size: int = 1,
                 model_retrain_interval: int = 20,
                 n_parallel_jobs: int = 5):
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
        self.time_prediction = time_prediction
        self.feature_manager = feature_manager
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.step_size = step_size
        self.n_parallel_jobs = n_parallel_jobs
        self.model_retrain_interval = model_retrain_interval
        self.initial_run = True
        self.model_retrain_counter = 0
        self.model = model
        self.persist_samples = persist_samples
        self.feature_importance = None

        # set up sample directory if true
        if self.persist_samples:
            # create the samples directory
            self.sample_dir = Path(f'results/{self.time_prediction}/samples/dgs_yields')
            Path(self.sample_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.sample_dir = None
        
        # Results storage
        self.results_list = []
        # Model tracking
        self.model_summaries: List[Dict] = []
        
        logger.info(f"Initialized walk-forward validator: window_size={window_size}, "
                   f"min_window_size={min_window_size}, step_size={step_size}")

    
    def run_single_prediction(self, 
                            train_start_idx: int,
                            train_end_idx: int, 
                            predict_idx: int,
                            target_columns: list[str],
                            features: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run a single prediction step in walk-forward validation.
        
        Args:
            train_start_idx: Start index of training window
            train_end_idx: End index of training window  
            predict_idx: Index to make prediction for
            target_columns: list[str]: Target variable names
            bond_index: Bond index name
            features: List of feature names
            
        Returns:
            Dictionary with prediction results
        """
        # Get training data
        x_train, y_train = self.data_loader.get_window_data(
            start_idx=train_start_idx,
            end_idx=train_end_idx,
            target_columns=target_columns,
            feature_columns=features
        )
        # Check if a retrain is needed
        if self.model_retrain_counter == self.model_retrain_interval or self.initial_run:
            retrain = True
            logger.info(f"Running retrain for window {x_train.index.min()}-{x_train.index.max()}")
            # train the model
            self.model.train_historical(x=x_train, y=y_train)
            self.feature_importance = self.model.get_feature_importance_proxy(X=x_train)
            # Reset counter
            self.model_retrain_counter = 0
            if self.initial_run:
                logger.info("Initial run completed, model retraining interval reached")
                self.initial_run = False
        else:
            retrain = False

        # Get prediction features
        x_predict = self.data_loader.get_prediction_point(idx=predict_idx, feature_columns=features)

        # Get actual value
        actual_value = self.data_loader.get_actual_value(idx=predict_idx, target_columns=target_columns)

        # Get prediction date
        predict_date = self.data_loader.get_date_for_index(predict_idx)
        # Make prediction
        prediction_val = self.model.predict_val(x=x_predict)
        # Make prediciton samples

        prediction_samples = self.model.predict_val_distribution(x=x_predict, y=y_train, n_samples=1000)


        # Persist samples if needed
        if self.persist_samples:
            # Convert the 1, 4, 1000 array to a dataframe and save as parquet

            samples_df = pd.DataFrame(prediction_samples.reshape(-1, prediction_samples.shape[-1]).T,
                                      columns=target_columns)

            samples_df.to_parquet(self.sample_dir / f"{predict_date}.parquet")
        if isinstance(self.model, GaussianProcessEnsemble):
            result = {
                'date': predict_date,
                'actual_value': list(actual_value.to_numpy()),
                'prediction': list(prediction_val),
                # 'prediction_std': list(prediction_std),
                'best_kernel': self.model.best_kernel_name,
                'retrain': retrain,
            }
        else:  # BayesianRidgeEnsemble
            result = {
                'date': predict_date,
                'actual_value': list(actual_value.to_numpy()),
                'prediction': list(prediction_val),
                # 'prediction_std': list(prediction_std),
                'best_alpha': self.model.best_alpha_name,
                'retrain': retrain,
            }
        model_summary = self.model.get_model_summary()
        model_summary['date'] = predict_date
        # get the feature importance
        model_summary['feature_importance'] = self.feature_importance.to_dict()
        # Increment retrain counter
        self.model_retrain_counter += 1


        logger.debug(f"Prediction completed for {predict_date}: "
                    f"actual={actual_value}, pred={prediction_val}")

        return result, model_summary

    
    def run_walk_forward_validation(self) -> None:
        """
        Run complete walk-forward validation.
        
        Args:
            bond_index: Bond index to forecast
            max_predictions: Maximum number of predictions (for testing)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary with complete validation results
        """
        logger.info(f"Starting walk-forward validation for {self.time_prediction}")
        
        # Get features
        features = self.feature_manager.features_config[self.time_prediction]['max_features']
        target_columns = self.feature_manager.get_dependent_variables()
        
        # Get time windows
        windows = self.data_loader.get_time_windows(window_size=self.window_size, min_window_size=self.min_window_size)[-15:]
        logger.info(f"Running {len(windows)} predictions with {len(features)} features")
        # Run predictions
        for i, (train_start_idx, train_end_idx) in enumerate(windows):
            predict_idx = train_end_idx
            
            # Safety check for prediction index
            if predict_idx >= len(self.data_loader.data):
                logger.warning(f"Prediction index {predict_idx} out of bounds, stopping")
                break
            
            # Run single prediction
            result, model_summary = self.run_single_prediction(
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                predict_idx=predict_idx,
                target_columns=target_columns,
                features=features
            )
            # Store result
            self.results_list.append(result)
            self.model_summaries.append(model_summary)

    
    def export_results(self, filepath: str) -> None:
        """
        Export validation results to file.
        
        Args:
            filepath: Path to save results
        """

        results_df = pd.DataFrame(self.results_list)
        model_summary_df = pd.DataFrame(self.model_summaries)
        # create file path if not exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        try:
            results_df.to_parquet(f'{filepath}/{self.time_prediction}_dgs_results.parquet', index=False)
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
        try:
            model_summary_df.to_parquet(f'{filepath}/{self.time_prediction}_model_summary.parquet', index=False)
            logger.info(f"Results exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting model summary: {str(e)}")

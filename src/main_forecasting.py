"""
Main script for bond yield forecasting using Gaussian Process regression
with walk-forward validation and parallel kernel selection.

This script orchestrates the entire forecasting pipeline including:
- Data loading and feature selection
- Walk-forward validation
- Parallel kernel training and selection
- Performance evaluation and visualization
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

# Add the src directory to path to import our modules


from data_loader import BondDataLoader
from feature_manager import FeatureManager
from gp_models import GaussianProcessEnsemble
from bayesian_ridge_models import BayesianRidgeEnsemble
from walk_forward import WalkForwardValidator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecasting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YieldForecastingPipeline:
    """Main pipeline class for bond yield forecasting."""
    
    def __init__(self, model_name: str, time_prediction: str, config_file: str, data_file:str,
                 train_window: int, min_train_window: int, retrain_interval: int,
                 selection_metric: str= 'train_cosine_distance') -> None:
        """
        Initialize the forecasting pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing all parameters
        """
        self.time_prediction = time_prediction
        self.train_window = train_window
        self.min_train_window = min_train_window
        self.retrain_interval = retrain_interval
        # initialize components
        self.feature_manager = FeatureManager(features_config_path=config_file)
        self.data_loader = BondDataLoader(data_path=data_file)
        # check if model_name is valid
        if model_name not in ['GP', 'BayesianRidge']:
            raise ValueError("model_name must be either 'GP' or 'BayesianRidge'")

        self.model_obj = GaussianProcessEnsemble(selection_metric=selection_metric) if model_name == 'GP' else BayesianRidgeEnsemble(selection_metric=selection_metric)
        logger.info(
            f"Initialized forecasting pipeline for time prediction {self.time_prediction} "
            f"with train window {self.train_window}, min train window {self.min_train_window}, "
            f"retrain interval {self.retrain_interval}, and selection metric {selection_metric}"
        )
        
    def run_pipeline(self) -> None:
        """
        Run the complete forecasting pipeline.
        Returns:

        """
        logger.info("Starting yield forecasting pipeline")

        # Get features for the specified time prediction
        features = self.feature_manager.get_features_for_time_pred(time_prediction=self.time_prediction)
        target_columns = self.feature_manager.get_dependent_variables()
        self.data_loader.load_data(x=features, y=target_columns)
        logger.info(f"Using {len(features)} features for time prediction {self.time_prediction}")

        # Set up walk-forward validator
        wf_validator = WalkForwardValidator(
            model=self.model_obj,
            data_loader = self.data_loader,
            feature_manager = self.feature_manager,
            time_prediction=self.time_prediction,
            window_size=self.train_window,
            min_window_size=self.min_train_window,
            model_retrain_interval=self.retrain_interval
        )

        # Execute walk-forward validation
        wf_validator.run_walk_forward_validation()
        wf_validator.export_results(filepath=f'results/{self.time_prediction}/')

        logger.info("Yield forecasting pipeline completed")

if __name__ == "__main__":
    start_time = time.time()
    # time_prediction = 'seven-day-ahead'
    config_file = 'data/features_selected.yaml'
    # data_file = "/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/bond_yields_ns_params_shifted_7.parquet"
    train_window = 300
    min_train_window = 200
    retrain_interval = 7
    selection_metric = 'train_r2_avg'
    file_num = [1, 7, 30, 60]
    time_prediction_list = [
        'one-day-ahead', 'seven-day-ahead', 'thirty-day-ahead', 'sixty-day-ahead'
    ]
    for day, time_prediction in zip(file_num, time_prediction_list):
        data_file = f"data/bond_yields_ns_params_shifted_{day}.parquet"
        print(f"Running pipeline for {time_prediction} using data file {data_file}")
        pipeline = YieldForecastingPipeline(
            model_name='GP',
            time_prediction=time_prediction,
            config_file=config_file,
            data_file=data_file,
            train_window=train_window,
            min_train_window=min_train_window,
            retrain_interval=retrain_interval,
            selection_metric=selection_metric
        )
        pipeline.run_pipeline()

    end_time = time.time()
    # convert time to hours
    hours = (end_time - start_time) / 3600
    logger.info(f"Total execution time hours: {hours:.2f}")
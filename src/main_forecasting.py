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
    
    def __init__(self, time_prediction: str, config_file: str, data_file:str,
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
        self.gp_model = GaussianProcessEnsemble(selection_metric=selection_metric)
        
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
            model=self.gp_model,
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
    time_prediction = 'seven-day-ahead'
    config_file = '/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/features_selected.yaml'
    data_file = "/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/bond_yields_ns_params_shifted_7.parquet"
    train_window = 2000
    min_train_window = 1000
    retrain_interval = 30
    selection_metric = 'train_cosine_distance'
    pipeline = YieldForecastingPipeline(
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
"""
Main script for bond yield forecasting using Monte Carlo simulation
with walk-forward validation and distribution-based prediction.

This script orchestrates the entire simulation pipeline including:
- Data loading and feature selection for simulation
- Walk-forward validation with Monte Carlo simulation
- Distribution fitting and sampling for yield prediction
- Performance evaluation and sample persistence
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd

# Import simulation components
from data_loader import BondDataLoader
from feature_manager_sim import FeatureManagerSim
from walk_forward_sim import WalkForwardSimulator
from monte_sim import MonteCarloSimulator


# Configure logging for simulation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_forecasting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YieldSimulationPipeline:
    """Main pipeline class for Monte Carlo simulation-based bond yield forecasting."""
    
    def __init__(self, time_prediction: str, config_file: str, data_file: str,
                 train_window: int, min_train_window: int, 
                 n_simulations: int = 1000,
                 persist_samples: bool = True) -> None:
        """
        Initialize the simulation pipeline with configuration.
        
        Args:
            time_prediction: Time horizon for prediction (e.g., 'one-day-ahead')
            config_file: Path to features configuration YAML file
            data_file: Path to bond yield data parquet file
            train_window: Size of training window for simulation
            min_train_window: Minimum training window size
            n_simulations: Number of Monte Carlo simulations per prediction
            persist_samples: Whether to save individual simulation samples
        """
        self.time_prediction = time_prediction
        self.train_window = train_window
        self.min_train_window = min_train_window
        self.n_simulations = n_simulations
        self.persist_samples = persist_samples
        self.data_file = data_file
        
        # Initialize simulation components
        self.feature_manager_sim = FeatureManagerSim(features_config_path=config_file)
        self.data_loader = BondDataLoader(data_path=data_file)
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(
            f"Initialized simulation pipeline for time prediction {self.time_prediction} "
            f"with train window {self.train_window}, min train window {self.min_train_window}, "
            f"and {self.n_simulations} simulations per prediction"
        )
        
    def _validate_configuration(self) -> None:
        """Validate simulation configuration."""
        if self.n_simulations < 100:
            logger.warning("Low number of simulations may produce unstable results")
        
        if self.train_window < 1000:
            logger.warning("Small training window may not capture distribution properly")
        
        # Validate data file exists
        if not Path(self.data_file).exists():
            raise FileNotFoundError(f"Simulation data file not found: {self.data_file}")
        
        # Validate time prediction is supported
        available_times = self.feature_manager_sim.get_all_available_times()
        if self.time_prediction not in available_times:
            raise ValueError(f"Time prediction '{self.time_prediction}' not supported. "
                           f"Available: {available_times}")
        
    def run_simulation_pipeline(self) -> None:
        """
        Run the complete Monte Carlo simulation pipeline.
        """
        logger.info("Starting yield simulation pipeline")

        # Get features for the specified time prediction
        features = self.feature_manager_sim.get_features_for_time_pred(time_prediction=self.time_prediction)
        y_variables = self.feature_manager_sim.get_dependent_variables(self.time_prediction)
        target_columns = self.feature_manager_sim.get_target_variables()
        actual_columns = self.feature_manager_sim.get_actual_variables()
        
        # Load data with all required columns
        all_columns = features + y_variables + target_columns + actual_columns
        self.data_loader.load_data(x=features, y=y_variables + target_columns, actuals=actual_columns)
        
        logger.info(f"Using {len(features)} features for time prediction {self.time_prediction}")
        logger.info(f"Simulating {len(y_variables)} dependent variables")
        logger.info(f"Targeting {len(target_columns)} yield values")

        # Set up walk-forward simulator
        wf_simulator = WalkForwardSimulator(
            data_loader=self.data_loader,
            feature_manager_sim=self.feature_manager_sim,
            time_prediction=self.time_prediction,
            window_size=self.train_window,
            min_window_size=self.min_train_window,
            n_simulations=self.n_simulations,
            persist_samples=self.persist_samples
        )

        # Execute walk-forward simulation
        logger.info("Starting walk-forward simulation validation")
        simulation_start_time = time.time()
        
        wf_simulator.run_walk_forward_validation()
        
        simulation_end_time = time.time()
        simulation_duration = (simulation_end_time - simulation_start_time) / 60  # minutes
        logger.info(f"Simulation completed in {simulation_duration:.2f} minutes")
        
        # Export results
        results_dir = f'results/{self.time_prediction}/simulation/'
        wf_simulator.export_results(filepath=results_dir)
        
        # Log simulation statistics
        if wf_simulator.results_list:
            n_predictions = len(wf_simulator.results_list)
            total_simulations = n_predictions * self.n_simulations
            logger.info(f"Generated {n_predictions} predictions with {total_simulations} total simulations")
            
            if self.persist_samples:
                samples_dir = f'results/{self.time_prediction}/samples/monte_carlo'
                logger.info(f"Simulation samples saved to {samples_dir}")

        logger.info("Yield simulation pipeline completed")


if __name__ == "__main__":
    start_time = time.time()
    
    # Simulation-specific configuration
    config_file = 'data/features_selected4.yaml'  # Use simulation config
    train_window = 800  # Larger window for better distribution fitting
    min_train_window = 600  # Larger minimum window
    n_simulations = 1000  # Number of Monte Carlo simulations
    persist_samples = True  # Save simulation samples for analysis
    
    # File mappings for different time horizons
    file_num = [1, 7, 30, 60]
    time_prediction_list = [
        'one-day-ahead', 'seven-day-ahead', 'thirty-day-ahead', 'sixty-day-ahead'
    ]
    
    logger.info("Starting Monte Carlo simulation forecasting pipeline")
    logger.info(f"Configuration: train_window={train_window}, min_train_window={min_train_window}")
    logger.info(f"Simulation settings: n_simulations={n_simulations}, persist_samples={persist_samples}")
    
    for day, time_prediction in zip(file_num, time_prediction_list):
        data_file = f"data/bond_yields_train_shifted_{day}_sim.parquet"  # Use simulation data files
        
        logger.info(f"Running simulation pipeline for {time_prediction} using data file {data_file}")
        
        try:
            pipeline = YieldSimulationPipeline(
                time_prediction=time_prediction,
                config_file=config_file,
                data_file=data_file,
                train_window=train_window,
                min_train_window=min_train_window,
                n_simulations=n_simulations,
                persist_samples=persist_samples
            )
            
            pipeline.run_simulation_pipeline()
            logger.info(f"Successfully completed simulation for {time_prediction}")
            
        except Exception as e:
            import traceback
            logger.error(f"Error running simulation for {time_prediction}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            continue

    end_time = time.time()
    hours = (end_time - start_time) / 3600
    logger.info(f"Total simulation execution time: {hours:.2f} hours")
    logger.info("Monte Carlo simulation forecasting pipeline completed")
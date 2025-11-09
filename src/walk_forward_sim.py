"""
Walk-forward simulation framework for bond yield forecasting using Monte Carlo methods.
Implements sliding window validation with Monte Carlo simulation instead of ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import gc
from data_loader import BondDataLoader
from feature_manager_sim import FeatureManagerSim
from monte_sim import MonteCarloSimulator
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class WalkForwardSimulator:
    """Walk-forward simulation framework for time series forecasting using Monte Carlo methods."""
    
    def __init__(self,
                 data_loader: BondDataLoader,
                 feature_manager_sim: FeatureManagerSim,
                 time_prediction: str,
                 persist_samples: bool = True,
                 window_size: int = 3000,
                 min_window_size: int = 2000,
                 step_size: int = 1,
                 n_simulations: int = 1000):
        """
        Initialize walk-forward simulator.
        
        Args:
            data_loader: Configured data loader
            feature_manager_sim: Feature manager instance for simulation
            time_prediction: Time horizon for prediction (e.g., 'one-day-ahead')
            persist_samples: Whether to save simulation samples
            window_size: Size of training window
            min_window_size: Minimum window size for training
            step_size: Step size for walking forward
            n_simulations: Number of Monte Carlo simulations to run
        """
        self.data_loader = data_loader
        self.time_prediction = time_prediction
        self.feature_manager_sim = feature_manager_sim
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.step_size = step_size
        self.n_simulations = n_simulations
        self.persist_samples = persist_samples

        # Set up sample directory if true
        if self.persist_samples:
            self.sample_dir = Path(f'results/{self.time_prediction}/samples/monte_carlo')
            Path(self.sample_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.sample_dir = None
        
        # Results storage
        self.results_list = []
        
        logger.info(f"Initialized walk-forward simulator: window_size={window_size}, "
                   f"min_window_size={min_window_size}, step_size={step_size}, "
                   f"n_simulations={n_simulations}")

    
    def run_single_prediction(self, 
                            train_start_idx: int,
                            train_end_idx: int, 
                            predict_idx: int,
                            y_variables: List[str],
                            target_columns: List[str],
                            features: List[str]) -> Dict[str, Any]:
        """
        Run a single prediction step using Monte Carlo simulation.
        
        Args:
            train_start_idx: Start index of training window
            train_end_idx: End index of training window  
            predict_idx: Index to make prediction for
            target_columns: Target variable names
            features: List of feature names
            
        Returns:
            Dictionary with prediction results
        """
        # Get training data (dependent variables for KDE fitting)
        x_train, y_train = self.data_loader.get_window_data(
            start_idx=train_start_idx,
            end_idx=train_end_idx,
            target_columns=y_variables,  # DGS*_gain_loss columns
            feature_columns=features
        )

        # Get actual future values for comparison
        actual_value = self.data_loader.get_actual_value(idx=predict_idx, target_columns=target_columns)  # DGS*_future_val columns

        # Get prediction date
        predict_date = self.data_loader.get_date_for_index(predict_idx)
        
        # Initialize prediction storage
        predictions = {}
        prediction_samples = {}
        mean_changes_dict = {}
        std_changes_dict = {}
        
        # For each dependent variable (gain_loss), fit KDE and simulate
        for dep_var in y_variables:
            # Extract historical changes for this yield
            historical_changes = y_train[dep_var].values
            
            # Get current yield value (base for adding changes)
            # Extract base yield name (e.g., 'DGS1MO' from 'DGS1MO_gain_loss_1d')
            base_yield_name = dep_var.split('_gain_loss_')[0]
            current_yield = self.data_loader.get_current_yield_value(predict_idx, base_yield_name)
            
            # Create Monte Carlo simulator
            mc_sim = MonteCarloSimulator(num_simulations=self.n_simulations)
            
            # Fit and simulate
            simulated_yields = mc_sim.fit_and_simulate(historical_changes, current_yield)
            
            # Store mean prediction and all samples
            predictions[dep_var] = np.mean(simulated_yields)
            # Ensure consistent flattening regardless of input shape
            flattened_samples = np.atleast_1d(simulated_yields).flatten()
            prediction_samples[dep_var] = flattened_samples
            
            # Store simulation statistics
            mean_changes_dict[dep_var] = mc_sim.mean
            std_changes_dict[dep_var] = mc_sim.standard_dev

        # Convert predictions to list format matching target columns order
        # Map dependent variables to target columns (same order)
        prediction_means = []
        for i, target_col in enumerate(target_columns):
            # Get corresponding dependent variable
            dep_var = y_variables[i] if i < len(y_variables) else list(predictions.keys())[i]
            prediction_means.append(predictions[dep_var])

        # Persist samples if needed
        if self.persist_samples:
            # Create samples DataFrame with all simulations for all yields
            # Map dependent variables to target column names for consistency
            samples_data = {}
            for i, dep_var in enumerate(y_variables):
                # Map DGS1MO_gain_loss_1d -> DGS1MO_future_val for sample storage
                target_col = target_columns[i] if i < len(target_columns) else dep_var
                samples_array = prediction_samples[dep_var]
                
                # Debug: Check array length
                logger.debug(f"Variable {dep_var} -> {target_col}: array shape {samples_array.shape}, length {len(samples_array)}")
                
                # Ensure all arrays have exactly n_simulations length
                if len(samples_array) != self.n_simulations:
                    logger.warning(f"Array length mismatch for {dep_var}: expected {self.n_simulations}, got {len(samples_array)}")
                    # Truncate or pad to correct length
                    if len(samples_array) > self.n_simulations:
                        samples_array = samples_array[:self.n_simulations]
                    else:
                        # Pad with last value if too short
                        pad_length = self.n_simulations - len(samples_array)
                        samples_array = np.append(samples_array, [samples_array[-1]] * pad_length)
                
                samples_data[target_col] = samples_array
            
            samples_df = pd.DataFrame(samples_data)
            # Use prediction date as filename for easy identification
            samples_df.to_parquet(self.sample_dir / f"{predict_date.strftime('%Y-%m-%d')}.parquet")

        # Create result dictionary
        result = {
            'date': predict_date,
            'actual_value': list(actual_value.to_numpy().flatten()),
            'prediction': prediction_means,
            'simulation_stats': {
                'mean_changes': mean_changes_dict,
                'std_changes': std_changes_dict,
                'num_simulations': self.n_simulations
            }
        }

        logger.debug(f"Simulation completed for {predict_date}: "
                    f"actual={actual_value.values.flatten()}, pred={prediction_means}")

        return result

    
    def run_walk_forward_validation(self) -> None:
        """
        Run complete walk-forward validation using Monte Carlo simulation.
        """
        logger.info(f"Starting walk-forward simulation for {self.time_prediction}")
        
        # Get features
        features = self.feature_manager_sim.get_features_for_time_pred(self.time_prediction)
        y_variables = self.feature_manager_sim.get_dependent_variables(self.time_prediction)
        target_columns = self.feature_manager_sim.get_target_variables()
        
        # Get time horizon offset to prevent data leakage
        time_offset_map = {
            'one-day-ahead': 1,
            'seven-day-ahead': 7,
            'thirty-day-ahead': 30,
            'sixty-day-ahead': 60
        }
        
        if self.time_prediction not in time_offset_map:
            raise ValueError(f"Invalid time_prediction: {self.time_prediction}")
        
        offset = time_offset_map[self.time_prediction]
        
        # Get time windows
        windows = self.data_loader.get_time_windows(window_size=self.window_size, min_window_size=self.min_window_size)
        
        # Clip windows to ensure prediction points exist and prevent data leakage
        windows = windows[:-offset]
        
        # Testing - comment out when running the full validation
        # windows = windows[:1]

        logger.info(f"Running {len(windows)} predictions with {len(features)} features")
        
        # Run predictions
        for i, (train_start_idx, train_end_idx) in enumerate(windows):
            # Prediction point is offset days after training end to prevent data leakage
            predict_idx = train_end_idx + offset
            
            # Safety check for prediction index
            if predict_idx >= len(self.data_loader.data):
                logger.warning(f"Prediction index {predict_idx} out of bounds, stopping")
                break
            
            # Run single prediction
            result = self.run_single_prediction(
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                predict_idx=predict_idx,
                y_variables=y_variables,
                target_columns=target_columns,
                features=features
            )
            
            # Store result
            self.results_list.append(result)

    
    def export_results(self, filepath: str) -> None:
        """
        Export simulation results to file.
        
        Args:
            filepath: Path to save results
        """
        results_df = pd.DataFrame(self.results_list)
        
        # Create file path if not exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            results_df.to_parquet(f'{filepath}/{self.time_prediction}_simulation_results.parquet', index=False)
            logger.info(f"Results exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
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
from sklearn.gaussian_process.kernels import DotProduct, ExpSineSquared, RBF, WhiteKernel

# Add the src directory to path to import our modules


from data_loader import BondDataLoader
from feature_manager import FeatureManager
from gp_models import GPModelManager
from walk_forward import WalkForwardValidator
from parallel_trainer import ParallelTrainer
from metrics import MetricsCalculator
from visualization import ForecastVisualizer, create_summary_report


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


class BondForecastingPipeline:
    """Main pipeline class for bond yield forecasting."""
    
    def __init__(self, config: Dict):
        """
        Initialize the forecasting pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing all parameters
        """
        self.config = config
        self.data_loader = BondDataLoader(config['data'])
        self.feature_manager = FeatureManager(config['features'])
        self.gp_manager = GPModelManager(config['kernels'])
        self.walk_forward = WalkForwardValidator(config['walk_forward'])
        self.parallel_trainer = ParallelTrainer(config['parallel'])
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = ForecastVisualizer()
        
        # Results storage
        self.results = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare the data for forecasting.
        
        Returns:
            Prepared DataFrame with features and target
        """
        logger.info("Loading and preparing data...")
        
        # Load main dataset
        data = self.data_loader.load_bond_data(
            self.config['data']['bond_data_path']
        )
        
        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        
        return data
    
    def run_single_bond_forecast(self, 
                               data: pd.DataFrame, 
                               bond_index: str) -> Dict:
        """
        Run forecasting for a single bond index.
        
        Args:
            data: Prepared data DataFrame
            bond_index: Name of the bond index to forecast
            
        Returns:
            Dictionary containing forecast results
        """
        logger.info(f"Starting forecast for {bond_index}...")
        
        # Get features for this bond index
        features = self.feature_manager.get_bond_features(bond_index)
        logger.info(f"Using {len(features)} features for {bond_index}")
        
        # Validate features exist in data
        available_features = self.feature_manager.validate_features(
            features, data.columns.tolist()
        )
        
        if len(available_features) == 0:
            logger.error(f"No valid features found for {bond_index}")
            return {}
        
        logger.info(f"Validated {len(available_features)} features")
        
        # Prepare feature matrix and target
        X = data[available_features].copy()
        y = data[bond_index].copy()
        
        # Remove rows with missing values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        dates = data.index[valid_mask]
        
        logger.info(f"Clean data shape: X={X.shape}, y={y.shape}")
        
        # Initialize results storage
        predictions = []
        actual_values = []
        confidence_intervals = []
        training_times = []
        best_kernels = []
        
        # Run walk-forward validation
        splits = self.walk_forward.create_splits(len(X))
        logger.info(f"Created {len(splits)} walk-forward splits")
        
        for i, (train_idx, test_idx) in enumerate(splits):
            if i % 10 == 0:
                logger.info(f"Processing split {i+1}/{len(splits)}")
            
            # Prepare training and test data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Skip if test set is empty or train set too small
            if len(X_test) == 0 or len(X_train) < self.config['walk_forward']['min_train_size']:
                continue
            
            start_time = time.time()
            
            # Train models in parallel for different kernels
            kernel_results = self.parallel_trainer.train_parallel_kernels(
                X_train, y_train, self.config['kernels']['available_kernels']
            )
            
            # Select best kernel based on cross-validation score
            best_kernel_name = max(kernel_results.keys(), 
                                 key=lambda k: kernel_results[k]['score'])
            best_model = kernel_results[best_kernel_name]['model']
            
            # Make prediction
            pred_mean, pred_std = best_model.predict(X_test, return_std=True)
            
            # Calculate confidence intervals (95%)
            confidence_level = self.config['forecasting'].get('confidence_level', 0.95)
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 99% CI
            
            ci_lower = pred_mean - z_score * pred_std
            ci_upper = pred_mean + z_score * pred_std
            
            training_time = time.time() - start_time
            
            # Store results
            predictions.extend(pred_mean.flatten())
            actual_values.extend(y_test.values.flatten())
            confidence_intervals.extend(list(zip(ci_lower.flatten(), ci_upper.flatten())))
            training_times.append(training_time)
            best_kernels.append(best_kernel_name)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        ci_lower = np.array([ci[0] for ci in confidence_intervals])
        ci_upper = np.array([ci[1] for ci in confidence_intervals])
        
        logger.info(f"Generated {len(predictions)} predictions for {bond_index}")
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            actual_values, predictions
        )
        
        # Additional statistics
        avg_training_time = np.mean(training_times)
        kernel_frequency = pd.Series(best_kernels).value_counts()
        
        # Prepare results dictionary
        results = {
            'bond_index': bond_index,
            'predictions': predictions,
            'actual': actual_values,
            'confidence_intervals': (ci_lower, ci_upper),
            'dates': dates[-len(predictions):],  # Align dates with predictions
            'metrics': metrics,
            'avg_training_time': avg_training_time,
            'total_training_time': sum(training_times),
            'kernel_frequency': kernel_frequency.to_dict(),
            'best_kernel_overall': kernel_frequency.index[0],
            'feature_count': len(available_features),
            'features_used': available_features
        }
        
        logger.info(f"Forecast completed for {bond_index}")
        logger.info(f"Metrics: {metrics}")
        
        return results
    
    def run_forecast_pipeline(self, bond_indices: List[str] = None) -> Dict:
        """
        Run the complete forecasting pipeline for specified bond indices.
        
        Args:
            bond_indices: List of bond indices to forecast. If None, uses all available.
            
        Returns:
            Dictionary containing results for all bond indices
        """
        logger.info("Starting bond forecasting pipeline...")
        
        # Load and prepare data
        data = self.load_and_prepare_data()
        
        # Determine which bond indices to forecast
        if bond_indices is None:
            available_bonds = [col for col in data.columns 
                             if col.startswith('DGS') and col in self.feature_manager.bond_features]
            bond_indices = available_bonds
        
        logger.info(f"Forecasting for bond indices: {bond_indices}")
        
        # Run forecasting for each bond index
        all_results = {}
        
        for bond_index in bond_indices:
            try:
                if bond_index not in data.columns:
                    logger.warning(f"Bond index {bond_index} not found in data, skipping...")
                    continue
                
                results = self.run_single_bond_forecast(data, bond_index)
                if results:
                    all_results[bond_index] = results
                    
            except Exception as e:
                logger.error(f"Error forecasting {bond_index}: {str(e)}")
                continue
        
        self.results = all_results
        logger.info(f"Pipeline completed for {len(all_results)} bond indices")
        
        return all_results
    
    def generate_reports_and_visualizations(self, output_dir: str = "results"):
        """
        Generate comprehensive reports and visualizations.
        
        Args:
            output_dir: Directory to save outputs
        """
        if not self.results:
            logger.warning("No results available for reporting")
            return
        
        logger.info(f"Generating reports and visualizations in {output_dir}...")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate summary report
        summary_data = {}
        for bond_index, results in self.results.items():
            summary_data[bond_index] = {
                **results['metrics'],
                'Best_Kernel': results['best_kernel_overall'],
                'Training_Time': results['avg_training_time'],
                'Predictions_Count': len(results['predictions']),
                'Feature_Count': results['feature_count']
            }
        
        summary_df = create_summary_report(
            summary_data, 
            os.path.join(output_dir, "summary_report.csv")
        )
        
        # Generate detailed results
        detailed_results = {}
        for bond_index, results in self.results.items():
            detailed_results[bond_index] = {
                'metrics': results['metrics'],
                'kernel_frequency': results['kernel_frequency'],
                'features_used': results['features_used'],
                'avg_training_time': results['avg_training_time'],
                'total_training_time': results['total_training_time']
            }
        
        # Save detailed results to JSON
        with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Generate visualizations for each bond
        for bond_index, results in self.results.items():
            bond_plots_dir = os.path.join(plots_dir, bond_index)
            
            self.visualizer.save_all_plots(
                save_dir=bond_plots_dir,
                dates=results['dates'],
                actual=results['actual'],
                predictions=results['predictions'],
                confidence_intervals=results['confidence_intervals'],
                metrics_dict=results['metrics'],
                kernel_results=results['kernel_frequency'],
                bond_name=bond_index
            )
        
        # Create comparative visualizations
        if len(self.results) > 1:
            # Metrics comparison across bonds
            metrics_comparison = {bond: results['metrics'] 
                                for bond, results in self.results.items()}
            
            fig = self.visualizer.plot_metrics_comparison(
                metrics_comparison, "Performance Comparison Across Bond Indices"
            )
            fig.savefig(os.path.join(plots_dir, "bond_comparison.png"), 
                       dpi=300, bbox_inches='tight')
            
        logger.info(f"Reports and visualizations saved to {output_dir}")


def create_default_config() -> Dict:
    """Create default configuration for the forecasting pipeline."""
    return {
        'data': {
            'bond_data_path': 'bond_train_diff.parquet',
            'features_config_path': 'bond_important_features.yaml'
        },
        'features': {
            'config_path': 'bond_important_features.yaml',
            'validate_features': True
        },
        'kernels': {
            'available_kernels': {
                'DotProduct': DotProduct(),
                'ExpSineSquared': ExpSineSquared(),
                'RBF': RBF(),
                'WhiteKernel': WhiteKernel()
            },
            'cv_folds': 3,
            'scoring': 'neg_mean_squared_error'
        },
        'walk_forward': {
            'window_size': 300,  # Trading days (1 year)
            'min_train_size': 252,
            'step_size': 1
        },
        'parallel': {
            'n_jobs': 4,  # Use all available cores
            'backend': 'joblib'
        },
        'forecasting': {
            'confidence_level': 0.95
        }
    }


def main():
    """Main function to run the forecasting pipeline."""
    parser = argparse.ArgumentParser(description='Bond Yield Forecasting with Gaussian Processes')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (JSON)')
    parser.add_argument('--bonds', nargs='+', default=None,
                       help='Bond indices to forecast (e.g., DGS10 DGS2)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to bond data file')
    parser.add_argument('--features-path', type=str, default=None,
                       help='Path to features configuration file')
    parser.add_argument('--window-size', type=int, default=252,
                       help='Walk-forward window size')
    parser.add_argument('--min-train-size', type=int, default=60,
                       help='Minimum training size')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    if args.data_path:
        config['data']['bond_data_path'] = args.data_path
    if args.features_path:
        config['data']['features_config_path'] = args.features_path
        config['features']['config_path'] = args.features_path
    if args.window_size:
        config['walk_forward']['window_size'] = args.window_size
    if args.min_train_size:
        config['walk_forward']['min_train_size'] = args.min_train_size
    if args.n_jobs:
        config['parallel']['n_jobs'] = args.n_jobs
    
    # Initialize and run pipeline
    try:
        pipeline = BondForecastingPipeline(config)
        
        # Run forecasting
        results = pipeline.run_forecast_pipeline(args.bonds)
        
        if results:
            # Generate reports and visualizations
            pipeline.generate_reports_and_visualizations(args.output_dir)
            
            # Print summary
            print("\n" + "="*60)
            print("FORECASTING SUMMARY")
            print("="*60)
            
            for bond_index, bond_results in results.items():
                metrics = bond_results['metrics']
                print(f"\n{bond_index}:")
                print(f"  MAE: {metrics['MAE']:.4f}")
                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
                print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
                print(f"  Best Kernel: {bond_results['best_kernel_overall']}")
                print(f"  Avg Training Time: {bond_results['avg_training_time']:.2f}s")
                print(f"  Features Used: {bond_results['feature_count']}")
            
            print(f"\nDetailed results saved to: {args.output_dir}")
            
        else:
            print("No successful forecasts generated.")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
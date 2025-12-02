"""
GP Modeling Experiments Class for bond yield forecasting.
Implements comprehensive kernel evaluation and experimentation framework.
"""

import numpy as np
import pandas as pd
import pickle
import logging
import warnings
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import (
    WhiteKernel, ConstantKernel, RBF, Matern, 
    RationalQuadratic, ExpSineSquared, DotProduct
)
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPModelingExperiments:
    """
    Comprehensive GP modeling experiments class for evaluating different kernels
    on bond yield forecasting datasets.
    """
    
    def __init__(self, 
                 selection_metric: str = 'train_cosine_distance',
                 random_state: int = 42,
                 n_jobs: int = 3,
                 normalize_features: bool = True,
                 normalize_targets: bool = False):
        """
        Initialize the GP modeling experiments.
        
        Args:
            selection_metric: Metric for kernel selection
            random_state: For reproducibility
            n_jobs: Number of parallel jobs
            normalize_features: Whether to standardize features
            normalize_targets: Whether to normalize target variables
        """
        # Validate selection metric
        valid_metrics = ['train_cosine_distance', 'train_euclidean_rmse', 'train_r2_avg', 'train_r2_flat']
        if selection_metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {selection_metric}. Available: {valid_metrics}")
        
        self.selection_metric = selection_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        
        # Class attributes
        self.raw_data: Optional[pd.DataFrame] = None
        self.prediction_columns: Optional[List[str]] = None
        self.feature_columns: Optional[List[str]] = None
        self.normalizer: Optional[StandardScaler] = None
        self.target_normalizer: Optional[StandardScaler] = None
        self.kernels: Dict[str, Any] = {}
        self.fitted_models: Dict[str, MultiOutputRegressor] = {}
        self.experiment_results: Dict[str, Dict[str, Any]] = {}
        self.best_kernel_name: Optional[str] = None
        self.best_model: Optional[MultiOutputRegressor] = None
        
        # Initialize kernel configurations
        self.setup_kernel_configurations()
        
        logger.info(f"Initialized GPModelingExperiments with {len(self.kernels)} kernels")

    def setup_kernel_configurations(self, custom_kernels: Dict[str, Any] = None) -> None:
        """
        Setup kernel configurations for experimentation.
        
        Args:
            custom_kernels: Optional dictionary of custom kernel configurations
        """
        default_kernels = {
            'RBF': (ConstantKernel(1.0, (1e-3, 1e3)) *
                   RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                   WhiteKernel(noise_level=1e-5)),
            
            'Matern_1.5': (ConstantKernel(1.0, (1e-3, 1e3)) *
                          Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
                          WhiteKernel(noise_level=1e-5)),
            
            'Matern_2.5': (ConstantKernel(1.0, (1e-3, 1e3)) *
                          Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
                          WhiteKernel(noise_level=1e-5)),
            
            'RationalQuadratic': (ConstantKernel(1.0, (1e-3, 1e3)) *
                                 RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                 length_scale_bounds=(1e-2, 1e2),
                                                 alpha_bounds=(1e-5, 1e5)) +
                                 WhiteKernel(noise_level=1e-5)),
            
            'ExpSineSquared': (ConstantKernel(1.0, (1e-3, 1e3)) *
                              ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                           length_scale_bounds=(1e-2, 1e2),
                                           periodicity_bounds=(1e-2, 1e2)) +
                              WhiteKernel(noise_level=1e-5)),
            
            'DotProduct': DotProduct() + WhiteKernel(noise_level=1e-5),
            
            'QuasiPeriodic': (ConstantKernel(1.0, (1e-3, 1e3)) *
                             RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) *
                             ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                          length_scale_bounds=(1e-2, 1e2),
                                          periodicity_bounds=(1e-2, 1e2)) +
                             WhiteKernel(noise_level=1e-5))
        }
        
        if custom_kernels:
            default_kernels.update(custom_kernels)
        
        self.kernels = default_kernels
        logger.info(f"Setup {len(self.kernels)} kernel configurations")

    def transform_features(self, x_features: pd.DataFrame) -> Tuple[StandardScaler, np.ndarray]:
        """
        Transform features using StandardScaler normalization.
        
        Args:
            x_features: DataFrame subset of features to be transformed
            
        Returns:
            Tuple of (fitted_scaler_object, transformed_features)
        """
        if not isinstance(x_features, pd.DataFrame):
            raise ValueError("x_features must be a pandas DataFrame")
        
        if self.normalize_features:
            scaler = StandardScaler()
            x_transformed = scaler.fit_transform(x_features)
            
            # Store the scaler for later use
            if self.normalizer is None:
                self.normalizer = scaler
            
            logger.debug(f"Transformed features shape: {x_transformed.shape}")
            return scaler, x_transformed
        else:
            # Return identity transformation
            return None, x_features.values

    def train_gp_model(self, 
                       x_features: pd.DataFrame, 
                       y_features: pd.Series, 
                       kernel: Any) -> Dict[str, Any]:
        """
        Train a Gaussian Process model with specified kernel.
        
        Args:
            x_features: Feature DataFrame
            y_features: Target pandas Series
            kernel: Defined kernel object
            
        Returns:
            Dictionary containing training results and metrics
        """
        if not isinstance(x_features, pd.DataFrame):
            raise ValueError("x_features must be a pandas DataFrame")
        if not isinstance(y_features, pd.Series):
            raise ValueError("y_features must be a pandas Series")
        
        # Apply feature transformation
        scaler, x_transformed = self.transform_features(x_features)
        
        # Handle target normalization if requested
        y_transformed = y_features.copy()
        target_scaler = None
        if self.normalize_targets:
            target_scaler = StandardScaler()
            y_transformed = pd.Series(
                target_scaler.fit_transform(y_features.values.reshape(-1, 1)).flatten(),
                index=y_features.index,
                name=y_features.name
            )
            if self.target_normalizer is None:
                self.target_normalizer = target_scaler
        
        # Create GP model
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,  # We handle normalization ourselves
            n_restarts_optimizer=3,
            random_state=self.random_state
        )
        
        # Check if we need MultiOutputRegressor
        if len(y_transformed.shape) == 1 or y_transformed.shape[1] == 1:
            # Single output case - use GP directly
            model = gp_model
            y_fit = y_transformed.values if hasattr(y_transformed, 'values') else y_transformed
        else:
            # Multi-output case - use MultiOutputRegressor
            model = MultiOutputRegressor(gp_model, n_jobs=self.n_jobs)
            y_fit = y_transformed
        
        # Fit model
        model.fit(x_transformed, y_fit)
        
        # Generate predictions on training data
        y_pred = model.predict(x_transformed)
        
        # Calculate residuals
        y_transformed_array = y_transformed.values if hasattr(y_transformed, 'values') else y_transformed
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_transformed_array.shape) == 1:
            y_transformed_array = y_transformed_array.reshape(-1, 1)
        residuals = y_transformed_array - y_pred
        
        # Extract log marginal likelihood
        if hasattr(model, 'estimators_'):
            # Multi-output case
            estimators = model.estimators_
            log_marginal_likelihood = [est.log_marginal_likelihood() for est in estimators]
        else:
            # Single output case
            log_marginal_likelihood = [model.log_marginal_likelihood()]
        
        # Calculate training metrics
        y_for_metrics = y_transformed_array
        training_metrics = self._calculate_metrics(y_for_metrics, y_pred)
        
        # Store the model and residuals
        self.model = model
        self.training_residuals = residuals
        
        results = {
            'kernel_name': str(kernel),
            'log_marginal_likelihood_value': log_marginal_likelihood,
            'training_residuals': residuals,
            'model': model,
            'training_metrics': training_metrics,
            'feature_scaler': scaler,
            'target_scaler': target_scaler
        }
        
        logger.info(f"Trained GP model with kernel: {str(kernel)[:50]}...")
        return results

    def predict(self, x_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x_features: Feature variables for prediction
            
        Returns:
            Dictionary containing predictions, std_deviations, and covariance
        """
        if self.model is None:
            raise ValueError("Must train a model first using train_gp_model")
        
        if not isinstance(x_features, pd.DataFrame):
            raise ValueError("x_features must be a pandas DataFrame")
        
        # Transform features using stored scaler
        if self.normalizer is not None:
            x_transformed = self.normalizer.transform(x_features)
        else:
            x_transformed = x_features.values
        
        # Generate predictions
        predictions = self.model.predict(x_transformed)
        
        # Handle uncertainty quantification
        std_deviations = []
        covariances = []
        
        if hasattr(self.model, 'estimators_'):
            # Multi-output case
            for i, estimator in enumerate(self.model.estimators_):
                if hasattr(estimator, 'predict'):
                    try:
                        pred_mean, pred_std = estimator.predict(x_transformed, return_std=True)
                        pred_cov = estimator.predict(x_transformed, return_cov=True)[1]
                        std_deviations.append(pred_std)
                        covariances.append(pred_cov)
                    except:
                        # Fallback if uncertainty estimation fails
                        std_deviations.append(np.zeros_like(predictions[:, i]))
                        covariances.append(np.eye(len(predictions)))
        else:
            # Single output case
            try:
                pred_mean, pred_std, pred_cov = self.model.predict(x_transformed, return_std=True, return_cov=True)
                # pred_cov = self.model.predict(x_transformed, return_cov=True)[1]
                std_deviations.append(pred_std)
                covariances.append(pred_cov)
            except:
                # Fallback if uncertainty estimation fails
                std_deviations.append(np.zeros_like(predictions.flatten()))
                covariances.append(np.eye(len(predictions)))
        
        # Inverse transform if target normalization was used
        if self.target_normalizer is not None:
            predictions = self.target_normalizer.inverse_transform(predictions)
        
        results = {
            'predictions': predictions,
            'std_deviations': np.array(std_deviations).T if std_deviations else None,
            'covariance': covariances
        }
        
        logger.debug(f"Generated predictions with shape: {predictions.shape}")
        return results

    def predict_samples(self, x_features: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate samples from the predictive distribution using residual sampling.
        
        Args:
            x_features: Feature variables for prediction
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with shape (n_samples, n_targets) containing prediction samples
        """
        if self.model is None:
            raise ValueError("Must train a model first using train_gp_model")
        
        if not hasattr(self, 'training_residuals') or self.training_residuals is None:
            raise ValueError("Training residuals not available. Train model first.")
        
        # Get mean predictions
        # prediction_results = self.predict(x_features)
        # y_mean = prediction_results['predictions']
        y_samples = self.model.sample_y(x_features.values, n_samples=n_samples, random_state=self.random_state)
        # convert y_samples to DataFrame
        samples_df = pd.DataFrame(y_samples, columns=self.prediction_columns)
        # check if y was normalized
        if self.target_normalizer is not None:
            samples_df = self.target_normalizer.inverse_transform(y_samples)
        return samples_df
        # # Bootstrap sample from training residuals
        # residuals = self.training_residuals
        # n_targets = residuals.shape[1] if len(residuals.shape) > 1 else 1
        #
        # # Ensure y_mean is 2D
        # if len(y_mean.shape) == 1:
        #     y_mean = y_mean.reshape(-1, 1)
        #
        # if n_targets == 1:
        #     # Single target case
        #     residuals_flat = residuals.flatten() if len(residuals.shape) > 1 else residuals
        #     # Generate samples for each prediction point
        #     y_samples = np.array([
        #         y_mean[i, 0] + np.random.choice(residuals_flat, size=n_samples, replace=True)
        #         for i in range(len(y_mean))
        #     ]).T  # Shape: (n_samples, n_prediction_points)
        # else:
        #     # Multi-target case
        #     y_samples = np.array([
        #         np.random.choice(residuals[:, col], size=n_samples, replace=True)
        #         for col in range(n_targets)
        #     ]).T
        #     y_samples = y_mean + y_samples
        #
        # # Create DataFrame with appropriate column names
        # if self.prediction_columns and len(self.prediction_columns) == y_samples.shape[1]:
        #     columns = self.prediction_columns
        # else:
        #     columns = [f'target_{i}' for i in range(y_samples.shape[1])]
        #
        # samples_df = pd.DataFrame(y_samples, columns=columns)
        #
        # logger.debug(f"Generated {n_samples} prediction samples")
        # return samples_df

    def fit_transform_predict_pipeline(self,
                                       x_train: pd.DataFrame,
                                       y_train: pd.Series,
                                       kernel: Any,
                                       x_predict: pd.DataFrame,
                                       samples: bool = True) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Optional[pd.DataFrame]]:
        """
        Execute complete pipeline: transform -> train -> predict -> sample.
        
        Args:
            x_train: Training feature DataFrame
            y_train: Training target Series
            kernel: Kernel object for GP model
            x_predict: Prediction feature DataFrame
            samples: Whether to generate prediction samples
            
        Returns:
            Tuple of (training_metrics, predict_metrics, y_samples)
        """
        logger.info("Starting fit-transform-predict pipeline")
        
        # Step 1: Transform features and train model
        training_metrics = self.train_gp_model(x_train, y_train, kernel)
        self.training_residuals = training_metrics['training_residuals']
        
        # Step 2: Generate predictions
        predict_metrics = self.predict(x_predict)
        
        # Step 3: Generate samples if requested
        y_samples = None
        if samples:
            y_samples = self.predict_samples(x_predict)
        
        logger.info("Completed fit-transform-predict pipeline")
        return training_metrics, predict_metrics, y_samples

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various training metrics."""
        metrics = {
            'train_cosine_distance': self._cosine_distance_avg(y_true, y_pred),
            'train_euclidean_rmse': self._euclidean_rmse_avg(y_true, y_pred),
            'train_r2_avg': self._rsquared_score_avg(y_true, y_pred),
            'train_r2_flat': self._rsquared_flat(y_true, y_pred)
        }
        return metrics

    def _cosine_distance_avg(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute average cosine distance between vectors."""
        if len(a.shape) == 1:
            a = a.reshape(-1, 1)
        if len(b.shape) == 1:
            b = b.reshape(-1, 1)
        
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        cosine_similarity = np.sum(a_norm * b_norm, axis=1)
        return np.mean(cosine_similarity)

    def _euclidean_rmse_avg(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean RMSE between vectors."""
        if len(a.shape) == 1:
            a = a.reshape(-1, 1)
        if len(b.shape) == 1:
            b = b.reshape(-1, 1)
        
        errors = np.linalg.norm(a - b, axis=1)
        return np.sqrt(np.mean(errors ** 2))

    def _rsquared_score_avg(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute average R-squared score."""
        scores = r2_score(a, b, multioutput='raw_values')
        return np.mean(scores)

    def _rsquared_flat(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute R-squared score on flattened arrays."""
        return r2_score(a.flatten(), b.flatten())

    def run_kernel_experiments(self, 
                              x_train: pd.DataFrame, 
                              y_train: pd.Series,
                              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Execute experiments across all configured kernels.
        
        Args:
            x_train: Training feature DataFrame
            y_train: Training target Series
            validation_data: Optional (x_val, y_val) tuple for validation
            
        Returns:
            Dictionary with comprehensive results for all kernels
        """
        logger.info(f"Starting kernel experiments with {len(self.kernels)} kernels")
        
        results = {}
        
        for kernel_name, kernel in self.kernels.items():
            logger.info(f"Training kernel: {kernel_name}")
            
            try:
                # Train the model
                training_result = self.train_gp_model(x_train, y_train, kernel)
                
                # Store the model for this kernel
                self.fitted_models[kernel_name] = training_result['model']
                
                # Prepare result dictionary
                kernel_result = {
                    'kernel_name': kernel_name,
                    'training_metrics': training_result['training_metrics'],
                    'log_marginal_likelihood': training_result['log_marginal_likelihood_value'],
                    'training_residuals': training_result['training_residuals'],
                    'model': training_result['model'],
                    'feature_scaler': training_result['feature_scaler'],
                    'target_scaler': training_result['target_scaler']
                }
                
                # Add validation metrics if validation data provided
                if validation_data is not None:
                    x_val, y_val = validation_data
                    val_predictions = self.predict(x_val)['predictions']
                    val_metrics = self._calculate_metrics(y_val.values, val_predictions)
                    kernel_result['validation_metrics'] = val_metrics
                
                results[kernel_name] = kernel_result
                
            except Exception as e:
                logger.error(f"Failed to train kernel {kernel_name}: {str(e)}")
                results[kernel_name] = {
                    'kernel_name': kernel_name,
                    'error': str(e),
                    'training_metrics': None
                }
        
        # Store results
        self.experiment_results = results
        
        # Select best model
        self.select_best_model()
        
        logger.info(f"Completed kernel experiments. Best kernel: {self.best_kernel_name}")
        return results

    def select_best_model(self, metric: Optional[str] = None) -> str:
        """
        Select the best model based on specified metric.
        
        Args:
            metric: Optional metric to use for selection. Uses class default if None
            
        Returns:
            Name of the best performing kernel
        """
        if not self.experiment_results:
            raise ValueError("Must run experiments first using run_kernel_experiments")
        
        metric = metric or self.selection_metric
        
        # Filter out failed experiments
        valid_results = {
            name: result for name, result in self.experiment_results.items()
            if 'error' not in result and result['training_metrics'] is not None
        }
        
        if not valid_results:
            raise ValueError("No valid experiment results found")
        
        # Select best based on metric
        if metric == 'train_cosine_distance':
            best_kernel = max(valid_results, 
                            key=lambda x: valid_results[x]['training_metrics']['train_cosine_distance'])
        elif metric == 'train_euclidean_rmse':
            best_kernel = min(valid_results,
                            key=lambda x: valid_results[x]['training_metrics']['train_euclidean_rmse'])
        elif metric == 'train_r2_avg':
            best_kernel = max(valid_results,
                            key=lambda x: valid_results[x]['training_metrics']['train_r2_avg'])
        elif metric == 'train_r2_flat':
            best_kernel = max(valid_results,
                            key=lambda x: valid_results[x]['training_metrics']['train_r2_flat'])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self.best_kernel_name = best_kernel
        self.best_model = valid_results[best_kernel]['model']
        self.model = self.best_model  # Set for compatibility
        
        # Store training residuals for sampling
        self.training_residuals = valid_results[best_kernel]['training_residuals']
        
        logger.info(f"Selected best kernel: {best_kernel} based on {metric}")
        return best_kernel

    def get_kernel_rankings(self) -> pd.DataFrame:
        """
        Get ranking table for all evaluated kernels.
        
        Returns:
            DataFrame with kernel performance rankings
        """
        if not self.experiment_results:
            raise ValueError("Must run experiments first using run_kernel_experiments")
        
        rankings_data = []
        
        for kernel_name, result in self.experiment_results.items():
            if 'error' not in result and result['training_metrics'] is not None:
                row = {
                    'kernel_name': kernel_name,
                    **result['training_metrics'],
                    'log_marginal_likelihood_mean': np.mean(result['log_marginal_likelihood'])
                }
                
                # Add validation metrics if available
                if 'validation_metrics' in result:
                    for key, value in result['validation_metrics'].items():
                        row[f'val_{key.replace("train_", "")}'] = value
                
                rankings_data.append(row)
        
        if not rankings_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(rankings_data)
        
        # Sort by the selection metric
        if self.selection_metric in ['train_cosine_distance', 'train_r2_avg', 'train_r2_flat']:
            df = df.sort_values(self.selection_metric, ascending=False)
        else:  # train_euclidean_rmse
            df = df.sort_values(self.selection_metric, ascending=True)
        
        return df.reset_index(drop=True)

    def save_experiment_results(self, filepath: str) -> None:
        """
        Save complete experiment state to file.
        
        Args:
            filepath: Path to save the experiment results
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving (exclude non-serializable objects)
        save_data = {
            'selection_metric': self.selection_metric,
            'random_state': self.random_state,
            'normalize_features': self.normalize_features,
            'normalize_targets': self.normalize_targets,
            'best_kernel_name': self.best_kernel_name,
            'feature_columns': self.feature_columns,
            'prediction_columns': self.prediction_columns,
            'kernel_names': list(self.kernels.keys()),
            'experiment_results_summary': {}
        }
        
        # Save experiment summaries (without models)
        for kernel_name, result in self.experiment_results.items():
            if 'error' not in result:
                summary = {
                    'kernel_name': kernel_name,
                    'training_metrics': result['training_metrics'],
                    'log_marginal_likelihood': result['log_marginal_likelihood']
                }
                if 'validation_metrics' in result:
                    summary['validation_metrics'] = result['validation_metrics']
                save_data['experiment_results_summary'][kernel_name] = summary
        
        # Save scalers if they exist
        if self.normalizer is not None:
            save_data['feature_scaler'] = self.normalizer
        if self.target_normalizer is not None:
            save_data['target_scaler'] = self.target_normalizer
        
        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved experiment results to {filepath}")

    def load_experiment_results(self, filepath: str) -> None:
        """
        Load experiment state from file.
        
        Args:
            filepath: Path to load the experiment results from
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Experiment file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore state
        self.selection_metric = save_data['selection_metric']
        self.random_state = save_data['random_state']
        self.normalize_features = save_data['normalize_features']
        self.normalize_targets = save_data['normalize_targets']
        self.best_kernel_name = save_data['best_kernel_name']
        self.feature_columns = save_data['feature_columns']
        self.prediction_columns = save_data['prediction_columns']
        
        # Restore scalers
        if 'feature_scaler' in save_data:
            self.normalizer = save_data['feature_scaler']
        if 'target_scaler' in save_data:
            self.target_normalizer = save_data['target_scaler']
        
        # Restore experiment results summary
        self.experiment_results = {}
        for kernel_name, summary in save_data['experiment_results_summary'].items():
            self.experiment_results[kernel_name] = summary
        
        # Recreate kernels
        self.setup_kernel_configurations()
        
        logger.info(f"Loaded experiment results from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the best fitted model.
        
        Returns:
            Dictionary with model summary information
        """
        if self.best_model is None:
            return {"status": "No model fitted"}
        
        summary = {
            "best_kernel": self.best_kernel_name,
            "selection_metric": self.selection_metric,
            "n_kernels_tested": len(self.experiment_results),
            "normalize_features": self.normalize_features,
            "normalize_targets": self.normalize_targets
        }
        
        if self.best_kernel_name and self.best_kernel_name in self.experiment_results:
            best_result = self.experiment_results[self.best_kernel_name]
            summary.update({
                "training_metrics": best_result['training_metrics'],
                "log_marginal_likelihood": best_result['log_marginal_likelihood']
            })
            
            if 'validation_metrics' in best_result:
                summary["validation_metrics"] = best_result['validation_metrics']
        
        return summary

    def plot_kernel_performance(self, save_path: Optional[str] = None) -> None:
        """
        Create visualizations for kernel performance comparison.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib and seaborn required for plotting")
            return
        
        df = self.get_kernel_rankings()
        if df.empty:
            logger.warning("No experiment results to plot")
            return
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Kernel Performance Comparison', fontsize=16)
        
        metrics = ['train_cosine_distance', 'train_euclidean_rmse', 'train_r2_avg', 'train_r2_flat']
        titles = ['Cosine Distance', 'Euclidean RMSE', 'R² Average', 'R² Flat']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if metric in df.columns:
                df_sorted = df.sort_values(metric, ascending=(metric == 'train_euclidean_rmse'))
                bars = ax.bar(range(len(df_sorted)), df_sorted[metric])
                ax.set_title(f'{title} by Kernel')
                ax.set_xlabel('Kernel')
                ax.set_ylabel(title)
                ax.set_xticks(range(len(df_sorted)))
                ax.set_xticklabels(df_sorted['kernel_name'], rotation=45, ha='right')
                
                # Highlight best performing kernel
                best_idx = 0 if metric != 'train_euclidean_rmse' else len(df_sorted) - 1
                bars[best_idx].set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.show()
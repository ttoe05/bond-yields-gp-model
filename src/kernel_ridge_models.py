"""
Kernel Ridge Regression ensemble for bond yield forecasting.
Implements multiple kernel configurations with time series cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import (
    WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct
)
from sklearn.metrics import mean_squared_error, r2_score
from math import floor
from scipy.stats import multivariate_normal
from base_ensemble_model import BaseEnsembleModel
from random import uniform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KernelRidgeEnsemble(BaseEnsembleModel):
    """Kernel Ridge Regression ensemble for multivariate bond yield prediction."""
    
    def __init__(self, 
                 training_metric: str = 'mse',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize the Kernel Ridge Regression ensemble.
        
        Args:
            training_metric: Metric for grid search CV ('mse', 'mse_flat', 'rmse', 'r2_avg', 'r2_flat')
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs for grid search
        """
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        
        self.training_metric = training_metric
        
        # Model attributes
        self.best_model: Optional[MultiOutputRegressor] = None
        self.kernels: List = []
        self.residuals: Optional[np.ndarray] = None
        self.feature_importance: Optional[pd.Series] = None
        self.best_kernel_name: Optional[str] = None
        self.best_alpha_name: Optional[float] = None
        
        # Create kernel configurations
        self.create_kernel_configuration()
        
        logger.info(f"Initialized KRR ensemble with {len(self.kernels)} kernels, "
                   f"metric={training_metric}, random_state={random_state}")

    def create_kernel_configuration(self) -> List:
        """
        Create list of available kernels based on analysis/kernel_regression.ipynb.
        
        Returns:
            List of kernel configurations
        """
        kernel_test = (ConstantKernel(1.0, (1e-3, 1e3)) *
                       # DotProduct() *
                       RBF(length_scale=1.5, length_scale_bounds=(1e-7, 1e7)) +
                       WhiteKernel(noise_level=1e-5))

        kernel_test2 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        # DotProduct() *
                        RationalQuadratic(length_scale=1.0, alpha=0.1,
                                          length_scale_bounds=(1e-5, 1e5),
                                          alpha_bounds=(1e-5, 1e5)) +
                        WhiteKernel(noise_level=1e-5))

        kernel_test3 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        DotProduct() +
                        RationalQuadratic(length_scale=1.0, alpha=0.1,
                                          length_scale_bounds=(1e-5, 1e5),
                                          alpha_bounds=(1e-5, 1e5)) +
                        WhiteKernel(noise_level=1e-5))

        kernel_test4 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        DotProduct() +
                        ExpSineSquared(length_scale=2, periodicity=20.0,
                                       length_scale_bounds=(0.01, 10),
                                       periodicity_bounds=(1e-2, 1e2)) +
                        WhiteKernel(noise_level=1e-5))

        kernel_test5 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        # DotProduct() *
                        ExpSineSquared(length_scale=2, periodicity=20.0,
                                       length_scale_bounds=(0.01, 10),
                                       periodicity_bounds=(1e-2, 1e2)) +
                        WhiteKernel(noise_level=1e-5))
        kernels = [
            # RBF Kernel
            kernel_test,
            kernel_test2,
            kernel_test3,
            kernel_test4,
            kernel_test5
        ]
        
        self.kernels = kernels
        logger.info(f"Created {len(kernels)} kernel configurations")
        return kernels

    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error."""
        return mean_squared_error(y_true, y_pred, multioutput='uniform_average')
    
    def calculate_mse_flat(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate flattened MSE across all outputs."""
        return mean_squared_error(y_true.ravel(), y_pred.ravel())
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate root mean squared error."""
        return np.sqrt(self.calculate_mse(y_true, y_pred))
    
    def calculate_r2_avg(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate average R² across outputs."""
        return r2_score(y_true, y_pred, multioutput='uniform_average')
    
    def calculate_r2_flat(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² on flattened predictions vs actuals."""
        return r2_score(y_true.ravel(), y_pred.ravel())

    def _get_scoring_function(self) -> str:
        """Get sklearn scoring function name for the training metric."""
        metric_mapping = {
            'mse': 'neg_mean_squared_error',
            'mse_flat': 'neg_mean_squared_error',  # Will handle in custom scorer if needed
            'rmse': 'neg_root_mean_squared_error',
            'r2_avg': 'r2',
            'r2_flat': 'r2'  # Will handle in custom scorer if needed
        }
        return metric_mapping.get(self.training_metric, 'neg_mean_squared_error')

    def train_krr(self, 
                  x_train: pd.DataFrame, 
                  y_train: pd.DataFrame, 
                  alpha_params: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Train kernel ridge regression using grid search time series cross validation.
        
        Args:
            x_train: Feature variables for training
            y_train: Multivariate dependent variables for prediction
            alpha_params: List of alpha parameters to test (default from notebook analysis)
            
        Returns:
            Dictionary with training metrics and best parameters
        """
        # Validate inputs using base class method
        self._validate_inputs(x_train, y_train)
        
        # Store training boundaries for prediction enforcement
        self._store_training_boundaries(y_train)
        
        if alpha_params is None:
            alpha_params = [1e-2, 1e0, 1e3, 1e4]  # From notebook analysis
        
        logger.debug(f"Training KRR with {len(self.kernels)} kernels and {len(alpha_params)} alpha values")
        
        # Create parameter grid
        param_grid = {
            'estimator__kernel': self.kernels,
            'estimator__alpha': alpha_params
        }
        
        # Create base model
        base_model = MultiOutputRegressor(KernelRidge())
        
        # Time series cross validation - adjust parameters based on data size
        # n_samples = len(x_train)
        # max_train_size = min(1000, n_samples // 2)
        # test_size = min(50, n_samples // 10)
        n_samples = len(x_train)
        if n_samples < 4000:
            n_splits = 3
            test_size = 60
        elif n_samples < 5000 and n_samples >= 4000:
            n_splits = 4
            test_size = 100
        else:
            n_splits = 5
            test_size = 150
        # # round down
        max_train_size = floor((n_samples - (test_size * 3)) / 3)
        # max_train_size = 1000
        # n_splits = 3
        # test_size = 100
        cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=self._get_scoring_function(),
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(x_train.values, y_train.values)
        
        # Store best model and parameters
        self.best_model = grid_search.best_estimator_
        self.best_kernel_name = str(grid_search.best_params_['estimator__kernel'])
        self.best_alpha_name = grid_search.best_params_['estimator__alpha']
        
        # Calculate training residuals
        train_predictions = self.best_model.predict(x_train.values)
        self.residuals = y_train.values - train_predictions
        
        # Mark as trained
        self.is_trained = True
        
        # Calculate training metrics
        metrics = {}
        if self.training_metric == 'mse':
            metrics['training_mse'] = self.calculate_mse(y_train.values, train_predictions)
        elif self.training_metric == 'mse_flat':
            metrics['training_mse_flat'] = self.calculate_mse_flat(y_train.values, train_predictions)
        elif self.training_metric == 'rmse':
            metrics['training_rmse'] = self.calculate_rmse(y_train.values, train_predictions)
        elif self.training_metric == 'r2_avg':
            metrics['training_r2_avg'] = self.calculate_r2_avg(y_train.values, train_predictions)
        elif self.training_metric == 'r2_flat':
            metrics['training_r2_flat'] = self.calculate_r2_flat(y_train.values, train_predictions)
        
        metrics.update({
            'best_kernel': self.best_kernel_name,
            'best_alpha': self.best_alpha_name,
            'best_cv_score': grid_search.best_score_,
            'n_features': x_train.shape[1],
            'n_samples': x_train.shape[0],
            'n_outputs': y_train.shape[1]
        })
        
        logger.debug(f"Training completed. Best alpha: {self.best_alpha_name}, "
                   f"Best kernel: {self.best_kernel_name[:50]}...")
        
        return metrics

    def predict_val(self, x: pd.DataFrame) -> np.ndarray:
        """
        Point prediction wrapper using the best model estimator.
        
        Args:
            x: Features for prediction
            
        Returns:
            Raw point predictions (boundaries applied in walk_forward)
        """
        self._validate_trained()
        self._validate_inputs(x)
        
        # Return raw predictions - boundaries applied in walk_forward after inverse transform
        return self.best_model.predict(x.values)

    def predict_val_distribution(self, 
                                x: pd.DataFrame, 
                                y: pd.DataFrame,
                                n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate prediction samples using multivariate normal distribution.
        
        Args:
            x: Features for prediction
            y: for labeling the columns in the DataFrame
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with prediction samples
        """
        self._validate_trained()
        self._validate_inputs(x)  # Only validate x, y is just for column reference
        
        if self.residuals is None:
            raise ValueError("Residuals not available. Ensure train_krr() was called.")
        
        # Get point prediction
        point_pred = self.predict_val(x)
        mean_prediction_df = pd.DataFrame(data=point_pred, columns=y.columns)
        residuals_df = pd.DataFrame(data=self.residuals, columns=y.columns)
        
        # individually sample the data by each column
        samples_dict = {}
        for pred_col in mean_prediction_df.columns:
            mean_val = mean_prediction_df[pred_col].values[0]
            std_val = residuals_df[pred_col].abs().mean()
            samples_dict[pred_col] = np.random.normal(
                loc=mean_val,
                scale=std_val,
                size=n_samples
            )
        
        # Convert to DataFrame with proper structure - boundaries applied in walk_forward
        samples_df = pd.DataFrame(samples_dict)
        
        return samples_df
    
    
    def _apply_yield_floor(self, values: np.ndarray) -> np.ndarray:
        """
        Apply non-negative floor to yield values (for inverse transforms).
        
        Args:
            values: Array of yield values that may contain negatives
            
        Returns:
            Values with negative yields set to small positive values
        """
        bounded_values = values.copy()
        negative_mask = bounded_values < 0
        
        if np.any(negative_mask):
            n_negative = np.sum(negative_mask)
            # get a random value between 0.1 and 0
            n_sample = uniform(0.001, 0.1)
            bounded_values[negative_mask] = n_sample  # 0.01%
            logger.debug(f"Applied yield floor to {n_negative} negative values")
            
        return bounded_values
    
    def apply_yield_constraints(self, predictions: np.ndarray, 
                              is_inverse_transformed: bool = False) -> np.ndarray:
        """
        Public method to apply yield constraints to predictions.
        Only applies negative yield floor - no upper bounds.
        
        Args:
            predictions: Prediction values to constrain
            is_inverse_transformed: Whether predictions are already inverse transformed
            
        Returns:
            Constrained predictions with yield floor applied
        """
        # Always apply only yield floor (no upper bounds)
        return self._apply_yield_floor(predictions)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of the trained model.
        
        Returns:
            Dictionary with model information
        """
        if self.best_model is None:
            return {'status': 'untrained'}
        
        summary = {
            'model_type': 'KernelRidgeEnsemble',
            'best_kernel': self.best_kernel_name,
            'best_alpha': self.best_alpha_name,
            'n_estimators': len(self.best_model.estimators_),
            'training_metric': self.training_metric,
            'random_state': self.random_state,
            'has_prediction_boundaries': self.training_percentiles is not None
        }
        
        # Add boundary information if available
        if self.training_percentiles is not None:
            summary['training_boundaries'] = self.training_percentiles
            
        return summary

    def get_feature_importance_proxy(self, X: pd.DataFrame) -> pd.Series:
        """
        Get feature importance proxy for KRR (using coefficient magnitudes).
        
        Args:
            X: Feature matrix
            
        Returns:
            Series with feature importance scores
        """
        if self.best_model is None:
            raise ValueError("Model must be trained first.")
        
        # For KRR, we can use the dual coefficients as a proxy for importance
        # This is an approximation since KRR doesn't have direct feature importance
        feature_importance = np.zeros(X.shape[1])
        
        for i, estimator in enumerate(self.best_model.estimators_):
            # Use the dual coefficients magnitude as importance proxy
            if hasattr(estimator, 'dual_coef_'):
                dual_coef_importance = np.abs(estimator.dual_coef_).mean()
                feature_importance += dual_coef_importance
        
        # Normalize
        feature_importance = feature_importance / len(self.best_model.estimators_)
        
        # Create series with feature names
        importance_series = pd.Series(feature_importance, index=X.columns)
        importance_series = importance_series.sort_values(ascending=False)
        
        self.feature_importance = importance_series
        return importance_series

    def train_historical(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Alias for train_krr to match interface with other models.
        
        Args:
            x: Feature variables for training
            y: Target variables for training
            
        Returns:
            Training metrics dictionary
        """
        return self.train_krr(x, y)
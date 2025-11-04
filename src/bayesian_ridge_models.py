"""
Bayesian Ridge regression models for bond yield forecasting.
Implements different alpha hyperparameters and model selection strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import BondDataLoader
from feature_manager import FeatureManager

import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianRidgeEnsemble:
    """Ensemble of Bayesian Ridge models with different alpha hyperparameters."""
    
    def __init__(self, selection_metric: str = 'train_cosine_distance', random_state: int = 42):
        """
        Initialize the Bayesian Ridge ensemble.
        
        Args:
            selection_metric: Metric to use for model selection
            random_state: Random state for reproducibility
        """
        metrics = ['train_cosine_distance', 'train_euclidean_rmse', 'train_r2_avg', 'train_r2_flat']
        if selection_metric not in metrics:
            raise ValueError(f"Unknown metric: {selection_metric}. Available metrics: {metrics}")
        
        self.selection_metric = selection_metric
        self.random_state = random_state
        self.alpha_configs = None
        self.fitted_models: Dict[str, MultiOutputRegressor] = {}
        self.model_scores: Dict[str, Dict[str, float]] = {}
        self.best_alpha_name: Optional[str] = None
        self.best_model: Optional[MultiOutputRegressor] = None
        self._create_alpha_configurations()

    def _create_alpha_configurations(self) -> Dict[str, float]:
        """
        Create different alpha configurations for testing.
        
        Returns:
            Dictionary mapping alpha names to alpha values
        """
        alphas = {
            'alpha_1e-6': 1e-6,
            'alpha_1e-4': 1e-4,
            'alpha_1e-2': 1e-2,
            'alpha_1': 1.0,
            'alpha_10': 10.0,
            'alpha_100': 100.0,
        }
        
        logger.info(f"Created {len(alphas)} alpha configurations")
        self.alpha_configs = alphas

    def create_bayesian_ridge_model(self, alpha_name: str) -> MultiOutputRegressor:
        """
        Create a Bayesian Ridge model with specified alpha.
        
        Args:
            alpha_name: Name of the alpha configuration to use
            
        Returns:
            Configured MultiOutputRegressor with BayesianRidge
        """
        if alpha_name not in self.alpha_configs:
            raise ValueError(f"Unknown alpha: {alpha_name}. "
                           f"Available alphas: {list(self.alpha_configs.keys())}")
        
        alpha_val = self.alpha_configs[alpha_name]
        
        bayesian_ridge = BayesianRidge(
            alpha_1=alpha_val,
            alpha_2=alpha_val,
            lambda_1=alpha_val,
            lambda_2=alpha_val,
            compute_score=True,
            fit_intercept=True,
            copy_X=True
        )
        
        model = MultiOutputRegressor(bayesian_ridge, n_jobs=1)
        
        return model

    def _cosine_distance_avg(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the cosine distance between two vectors."""
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        cosine_similarity = np.sum(a_norm * b_norm, axis=1)
        return np.mean(cosine_similarity)

    def _euclidean_rmse_avg(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the Euclidean RMSE between two vectors."""
        errors = np.linalg.norm(a - b, axis=1)
        return np.sqrt(np.mean(errors ** 2))

    def rsquared_score_avg(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the R-squared score between two vectors."""
        scores = r2_score(a, b, multioutput='raw_values')
        return np.mean(scores)

    def rsquared_flat(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the R-squared score between two flattened vectors."""
        return r2_score(a.flatten(), b.flatten())

    def train_historical(self, x: pd.DataFrame, y: pd.Series):
        """
        Train all Bayesian Ridge models with different alpha values.
        
        Args:
            x: Feature matrix
            y: Target matrix
        """
        logger.info(f"Training {len(self.alpha_configs)} Bayesian Ridge models")
        
        for alpha_name in self.alpha_configs.keys():
            model = self.create_bayesian_ridge_model(alpha_name=alpha_name)
            # get the standard deviation of the target variable
            target_std = np.std(y.to_numpy(), axis=0)
            # self.model_scores[alpha_name] = {'target_std': target_std}
            
            # Fit model
            model.fit(x, y)
            y_pred = model.predict(x)
            # get the residuals of the training data
            residuals = np.abs(y.to_numpy() - y_pred)
            # get the standard deviation of the residuals
            residual_mean = np.mean(residuals, axis=0)
            
            # Calculate metrics
            metrics = {
                'train_cosine_distance': self._cosine_distance_avg(y.to_numpy(), y_pred),
                'train_euclidean_rmse': self._euclidean_rmse_avg(y.to_numpy(), y_pred),
                'train_r2_avg': self.rsquared_score_avg(y.to_numpy(), y_pred),
                'train_r2_flat': self.rsquared_flat(y.to_numpy(), y_pred),
                'residual_mean': residual_mean,
                'target_std': target_std,
                'model': model
            }
            
            self.model_scores[alpha_name] = metrics
            logger.info(f"Trained {alpha_name}: R²={metrics['train_r2_avg']:.4f}, "
                       f"RMSE={metrics['train_euclidean_rmse']:.4f}")
        
        # Select best model
        self._select_best_model()
        logger.info(f"Best alpha: {self.best_alpha_name}")


    def _select_best_model(self) -> None:
        """
        Select the best model based on the selection metric.
        """
        if self.selection_metric == 'train_cosine_distance':
            self.best_alpha_name = max(self.model_scores, 
                                     key=lambda x: self.model_scores[x]['train_cosine_distance'])
        elif self.selection_metric == 'train_euclidean_rmse':
            self.best_alpha_name = min(self.model_scores, 
                                     key=lambda x: self.model_scores[x]['train_euclidean_rmse'])
        elif self.selection_metric == 'train_r2_avg':
            self.best_alpha_name = max(self.model_scores, 
                                     key=lambda x: self.model_scores[x]['train_r2_avg'])
        else:  # train_r2_flat
            self.best_alpha_name = max(self.model_scores, 
                                     key=lambda x: self.model_scores[x]['train_r2_flat'])

        self.best_model = self.model_scores[self.best_alpha_name]['model']

    def predict_val(self, x: pd.DataFrame, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Feature matrix for prediction
            return_std: Whether to return standard deviations
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        if self.best_model is None:
            raise ValueError("Must train models first")
        
        y_pred = self.best_model.predict(x)
        
        if return_std:
            # For Bayesian Ridge, we can estimate uncertainty using coefficient variance
            # This is a simplified approach - using prediction variance as proxy
            std_estimates = np.std(y_pred, axis=0, keepdims=True)
            std_estimates = np.repeat(std_estimates, y_pred.shape[0], axis=0)
            return y_pred, std_estimates
        else:
            return y_pred, None

    def predict_val_distribution(self, x: pd.DataFrame, y: pd.Series, n_samples: int = 1000) -> np.ndarray:
        """
        Generate samples from the predictive distribution.

        Args:
            x: Feature matrix for prediction
            n_samples: Number of samples to draw
            
        Returns:
            Array of shape (n_samples, n_outputs) for single prediction
        """
        if self.best_model is None:
            raise ValueError("Must train models first")
        
        y_pred = self.best_model.predict(x)[0]
        
        # For Bayesian Ridge, generate samples by adding noise to predictions
        # This is a simplified approach - in practice you'd use the posterior distribution
        # get the residual std deviation for the best model
        covar = y.cov()
        y_samples = np.random.multivariate_normal(y_pred, covar, n_samples)
        return y_samples.T[np.newaxis, :, :]


    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the fitted models.
        
        Returns:
            Dictionary with model summary information
        """
        if self.best_model is None:
            return {"status": "No model fitted"}
        
        summary = {
            "best_alpha": self.best_alpha_name,
            "best_alpha_value": self.alpha_configs[self.best_alpha_name],
            "model_scores": {'train_cosine_distance': self.model_scores[self.best_alpha_name]['train_cosine_distance'],
                             'train_euclidean_rmse': self.model_scores[self.best_alpha_name]['train_euclidean_rmse'],
                             'train_r2_avg': self.model_scores[self.best_alpha_name]['train_r2_avg'],
                             'train_r2_flat': self.model_scores[self.best_alpha_name]['train_r2_flat']},
            "residual_mean": self.model_scores[self.best_alpha_name]['residual_mean'],
            "available_alphas": list(self.alpha_configs.keys()),
            "n_features": None,
            "n_training_samples": None
        }
        
        # Try to get training data info if available
        if hasattr(self.best_model, 'estimators_') and len(self.best_model.estimators_) > 0:
            first_estimator = self.best_model.estimators_[0]
            if hasattr(first_estimator, 'coef_'):
                summary["n_features"] = len(first_estimator.coef_)
        
        return summary

    def get_feature_importance_proxy(self, X: pd.DataFrame) -> pd.Series:
        """
        Get feature importance using Bayesian Ridge coefficients.
        
        Args:
            X: Feature matrix used for training
            
        Returns:
            Series with feature importance scores
        """
        if self.best_model is None:
            raise ValueError("Must fit model first")
        
        # Get coefficients from all outputs and average them
        coefficients = []
        for estimator in self.best_model.estimators_:
            if hasattr(estimator, 'coef_'):
                coefficients.append(np.abs(estimator.coef_))
        
        if coefficients:
            # Average absolute coefficients across outputs
            avg_coefficients = np.mean(coefficients, axis=0)
            # Normalize to sum to 1
            importance_scores = avg_coefficients / np.sum(avg_coefficients)
            
            return pd.Series(importance_scores, index=X.columns, name='importance')
        
        # Fallback: uniform importance
        logger.warning("Could not extract feature importance, using uniform weights")
        uniform_importance = np.ones(X.shape[1]) / X.shape[1]
        return pd.Series(uniform_importance, index=X.columns, name='importance')
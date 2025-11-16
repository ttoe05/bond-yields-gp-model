"""
Bayesian Ridge regression models for bond yield forecasting.
Implements different alpha hyperparameters and model selection strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from base_ensemble_model import BaseEnsembleModel

import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianRidgeEnsemble(BaseEnsembleModel):
    """Ensemble of Bayesian Ridge models with different alpha hyperparameters."""
    
    def __init__(self, selection_metric: str = 'train_cosine_distance', random_state: int = 42, n_jobs: int = 3):
        """
        Initialize the Bayesian Ridge ensemble.
        
        Args:
            selection_metric: Metric to use for model selection
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        
        metrics = ['train_cosine_distance', 'train_euclidean_rmse', 'train_r2_avg', 'train_r2_flat']
        if selection_metric not in metrics:
            raise ValueError(f"Unknown metric: {selection_metric}. Available metrics: {metrics}")
        
        self.selection_metric = selection_metric
        self.alpha_configs = None
        self.fitted_models: Dict[str, MultiOutputRegressor] = {}
        self.model_scores: Dict[str, Dict[str, float]] = {}
        self.best_alpha_name: Optional[str] = None
        self.best_model: Optional[MultiOutputRegressor] = None
        self.residuals: Optional[np.ndarray] = None
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
        
        model = MultiOutputRegressor(bayesian_ridge, n_jobs=self.n_jobs)
        
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

    def train_historical(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all Bayesian Ridge models with different alpha values.
        
        Args:
            x: Feature DataFrame
            y: Target DataFrame
            
        Returns:
            Dictionary with training metrics
        """
        # Validate inputs using base class method
        self._validate_inputs(x, y)
        
        # Store training boundaries for prediction enforcement
        self._store_training_boundaries(y)
        
        logger.info(f"Training {len(self.alpha_configs)} Bayesian Ridge models")
        
        for alpha_name in self.alpha_configs.keys():
            model = self.create_bayesian_ridge_model(alpha_name=alpha_name)
            # get the standard deviation of the target variable
            target_std = np.std(y.to_numpy(), axis=0)
            
            # Fit model
            model.fit(x.values, y.values)
            y_pred = model.predict(x.values)
            # get the residuals of the training data
            residuals = np.abs(y.to_numpy() - y_pred)
            residual_vals = y.to_numpy() - y_pred
            # get the standard deviation of the residuals
            residual_mean = np.mean(residuals, axis=0)
            
            # Calculate metrics
            metrics = {
                'train_cosine_distance': self._cosine_distance_avg(y.to_numpy(), y_pred),
                'train_euclidean_rmse': self._euclidean_rmse_avg(y.to_numpy(), y_pred),
                'train_r2_avg': self.rsquared_score_avg(y.to_numpy(), y_pred),
                'train_r2_flat': self.rsquared_flat(y.to_numpy(), y_pred),
                'residual_mean': residual_mean,
                'residuals': residual_vals,
                'target_std': target_std,
                'model': model
            }
            
            self.model_scores[alpha_name] = metrics
        
        # Select best model
        self._select_best_model()
        
        # Store residuals for sampling
        self.residuals = self.model_scores[self.best_alpha_name]['residuals']
        
        self.is_trained = True
        logger.info(f"Best alpha: {self.best_alpha_name}")
        
        # Return training summary
        return {
            'best_alpha': self.best_alpha_name,
            'alpha_scores': self.model_scores[self.best_alpha_name],
            'n_alphas_tested': len(self.alpha_configs),
            'training_samples': len(x),
            'n_features': x.shape[1],
            'target_columns': list(y.columns)
        }


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

    def predict_val(self, x: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Feature matrix for prediction
            
        Returns:
            Array of predictions
        """
        self._validate_trained()
        self._validate_inputs(x)
        
        predictions = self.best_model.predict(x.values)
        
        # Apply prediction boundaries if available
        if self.training_columns is not None and len(predictions.shape) == 2:
            predictions = self._apply_prediction_boundaries(predictions, self.training_columns)
        
        return predictions




    def predict_val_distribution(self, x: pd.DataFrame, y: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate samples from the predictive distribution using residual-based sampling.
        Adopts the same methodology as KernelRidgeEnsemble for consistency.

        Args:
            x: Feature matrix for prediction
            y: Target DataFrame (for column names)
            n_samples: Number of samples to draw
            
        Returns:
            DataFrame with prediction samples
        """
        self._validate_trained()
        self._validate_inputs(x)  # Only validate x, y is just for column reference
        
        if self.residuals is None:
            raise ValueError("Residuals not available. Ensure train_historical() was called.")
        
        # Get point prediction
        point_pred = self.predict_val(x)
        mean_prediction_df = pd.DataFrame(data=point_pred, columns=y.columns)
        residuals_df = pd.DataFrame(data=self.residuals, columns=y.columns)
        
        # Sample using residual-based methodology (same as KRR)
        samples_dict = {}
        for pred_col in mean_prediction_df.columns:
            mean_val = mean_prediction_df[pred_col].values[0]
            # Use absolute mean of residuals as standard deviation
            std_val = residuals_df[pred_col].abs().mean()
            samples_dict[pred_col] = np.random.normal(
                loc=mean_val,
                scale=std_val,
                size=n_samples
            )
        
        samples_df = pd.DataFrame(samples_dict)
        
        # Apply boundaries if available
        if self.training_columns is not None:
            bounded_samples = self._apply_prediction_boundaries(samples_df.values, list(samples_df.columns))
            samples_df = pd.DataFrame(bounded_samples, columns=samples_df.columns)
        
        return samples_df


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
        # return pd.Series(uniform_importance, index=X.columns, name='importance')
"""
Gaussian Process regression models for bond yield forecasting.
Implements different kernel types and model selection strategies.
"""

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct, ExpSineSquared, RBF, WhiteKernel, 
    ConstantKernel, Matern, RationalQuadratic
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianProcessEnsemble:
    """Ensemble of Gaussian Process models with different kernels."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = 3):
        """
        Initialize the GP ensemble.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.kernels = None
        self.fitted_models: Dict[str, GaussianProcessRegressor] = {}
        self.kernel_scores: Dict[str, Dict[str, float]] = {}
        self.best_kernel_name: Optional[str] = None
        self.n_jobs = n_jobs
        self.best_model: Optional[GaussianProcessRegressor] = None
        self._create_kernel_configurations()


    def _create_kernel_configurations(self) -> Dict[str, Any]:
        """
        Create different kernel configurations for testing.
        
        Returns:
            Dictionary mapping kernel names to kernel objects
        """
        kernels = {
            'DotProduct': DotProduct() + WhiteKernel(noise_level=1e-5),
            
            'ExpSineSquared': (ConstantKernel(1.0, (1e-3, 1e3)) * 
                              ExpSineSquared(length_scale=1.0, periodicity=1.0, 
                                           length_scale_bounds=(1e-2, 1e2),
                                           periodicity_bounds=(1e-2, 1e2)) +
                              WhiteKernel(noise_level=1e-5)),
            
            'RBF': (ConstantKernel(1.0, (1e-3, 1e3)) * 
                   RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                   WhiteKernel(noise_level=1e-5)),
            
            # 'WhiteKernel': WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1)),
            
            # 'Matern': (ConstantKernel(1.0, (1e-3, 1e3)) *
            #           Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
            #           WhiteKernel(noise_level=1e-5)),
            
            # 'RationalQuadratic': (ConstantKernel(1.0, (1e-3, 1e3)) *
            #                      RationalQuadratic(length_scale=1.0, alpha=1.0,
            #                                      length_scale_bounds=(1e-2, 1e2),
            #                                      alpha_bounds=(1e-5, 1e5)) +
            #                      WhiteKernel(noise_level=1e-5)),
            
            # 'Combined_RBF_ExpSine': (ConstantKernel(1.0, (1e-3, 1e3)) *
            #                         RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) *
            #                         ExpSineSquared(length_scale=1.0, periodicity=1.0,
            #                                      length_scale_bounds=(1e-2, 1e2),
            #                                      periodicity_bounds=(1e-2, 1e2)) +
            #                         WhiteKernel(noise_level=1e-5))
        }
        
        logger.info(f"Created {len(kernels)} kernel configurations")
        self.kernels = kernels


    def create_gp_model(self, kernel_name: str, normalize_y: bool = False) -> GaussianProcessRegressor:
        """
        Create a Gaussian Process model with specified kernel.
        
        Args:
            kernel_name: Name of the kernel to use
            normalize_y: whether to normalize the dependent variable with 0 mean and unit variance
            
        Returns:
            Configured GaussianProcessRegressor
        """
        if kernel_name not in self.kernels:
            raise ValueError(f"Unknown kernel: {kernel_name}. "
                           f"Available kernels: {list(self.kernels.keys())}")
        
        kernel = self.kernels[kernel_name]
        
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Nugget parameter for numerical stability
            normalize_y=normalize_y,  # Normalize target values
            n_restarts_optimizer=3,  # Number of restarts for optimization
            random_state=self.random_state
        )
        
        return gp_model


    def ensemble_train_runner(self, task: tuple[pd.DataFrame, pd.Series, str, int]) -> Dict[str, Any]:
        """
        task: tuple
            x: feature matrix
            y: target vector
            kernel_name: name of the kernel
            cv_folds: number of folds for time series cross-validation
        Returns:
            GaussianProcessRegressor
        """
        x, y, kernel_name, cv_folds = task
        # create a copy of the data
        x_copy = x.copy()
        y_copy = y.copy()
        model = self.create_gp_model(kernel_name=kernel_name)

        # Use TimeSeriesSplit for time series data
        # cv = TimeSeriesSplit(n_splits=cv_folds)
        #
        # # Cross-validation scoring
        # cv_scores = cross_val_score(model, x_copy, y_copy, cv=cv,
        #                             scoring='neg_mean_squared_error')

        # Fit model to get additional metrics
        model.fit(x_copy, y_copy)
        y_pred = model.predict(x_copy)

        metrics = {
            # 'cv_mse': cv_scores.mean(),
            # 'cv_mse_std': cv_scores.std(),
            'kernel_name': kernel_name,
            'train_mse': mean_squared_error(y, y_pred),
            'train_mae': mean_absolute_error(y, y_pred),
            'log_marginal_likelihood': model.log_marginal_likelihood(),
            'model': model,
        }

        # logger.debug(f"Kernel {kernel_name} - CV MSE: {metrics['cv_mse']:.6f} "
        #              f"(±{metrics['cv_mse_std']:.6f})")
        # self.kernel_scores[kernel_name] = metrics
        # self.fitted_models[kernel_name] = model
        return metrics


    def train_kernels(self, x: pd.DataFrame, y: pd.Series, cv_folds: int = 3):
        """
        Train all the Gaussian Process models using the different kernels concurrently.
        Returns:
        """
        tasks = [(x, y, kernel_name, cv_folds) for kernel_name in self.kernels.keys()]
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self.ensemble_train_runner,task) for task in tasks]
            for future in tqdm(as_completed(futures), desc='Kernels trained:', total=len(futures)):
                res = future.result()
                results.append(res)
        # return the best model
        results.sort(key=lambda x: x['train_mse'], reverse=False)
        self.kernel_scores = {x['kernel_name']: {'train_mse': x['train_mse'], 'model': x['model']} for x in results}

    
    def _select_best_kernel(self) -> None:
        """
        Select the best kernel based on cross-validation performance.
        """
        logger.info("Evaluating kernel performance...")
        self.best_kernel_name = min(self.kernel_scores, key=lambda x: self.kernel_scores[x]['train_mse'])
        logger.info(f"Best kernel: {self.best_kernel_name}")
        self.best_model = self.kernel_scores[self.best_kernel_name]['model']

    
    def predict_val(self, x: pd.DataFrame, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Feature matrix for prediction
            return_std: Whether to return standard deviations
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        self._select_best_kernel()
        
        if return_std:
            y_pred, y_std = self.best_model.predict(x, return_std=True)
            return y_pred, y_std
        else:
            y_pred = self.best_model.predict(x, return_std=False)
            return y_pred, None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the fitted models.
        
        Returns:
            Dictionary with model summary information
        """
        if self.best_model is None:
            return {"status": "No model fitted"}
        
        summary = {
            "best_kernel": self.best_kernel_name,
            "best_kernel_params": str(self.best_model.kernel_),
            "log_marginal_likelihood": self.best_model.log_marginal_likelihood(),
            "kernel_scores": self.kernel_scores,
            "available_kernels": list(self.kernels.keys()),
            "n_features": self.best_model.X_train_.shape[1] if hasattr(self.best_model, 'X_train_') else None,
            "n_training_samples": self.best_model.X_train_.shape[0] if hasattr(self.best_model, 'X_train_') else None
        }
        
        return summary

    
    def get_feature_importance_proxy(self, X: pd.DataFrame) -> pd.Series:
        """
        Get a proxy for feature importance using kernel gradients.
        
        Args:
            X: Feature matrix used for training
            
        Returns:
            Series with feature importance scores
        """
        if self.best_model is None:
            raise ValueError("Must fit best model first")
        
        # For GP models, we can use the kernel's characteristic length scales
        # as a proxy for feature importance (smaller length scale = more important)
        
        if hasattr(self.best_model.kernel_, 'length_scale'):
            length_scales = self.best_model.kernel_.length_scale
            
            if hasattr(length_scales, '__len__') and len(length_scales) == X.shape[1]:
                # Invert length scales: smaller length scale = higher importance
                importance_scores = 1.0 / (length_scales + 1e-10)
                importance_scores = importance_scores / importance_scores.sum()
                
                return pd.Series(importance_scores, index=X.columns, name='importance')
        
        # Fallback: uniform importance
        logger.warning("Could not extract feature importance, using uniform weights")
        uniform_importance = np.ones(X.shape[1]) / X.shape[1]
        return pd.Series(uniform_importance, index=X.columns, name='importance')


if __name__ == "__main__":
    # load the data
    df_train = pd.read_parquet("bond_train_diff.parquet")
    logger.info(f"Loaded dataset with {len(df_train)} rows")
    # load the feature config yaml file
    with open("bond_important_features.yaml", 'r') as file:
        features_config = yaml.safe_load(file)
    target_var = list(features_config.keys())[0]
    logger.info(f"Target variable: {target_var}")
    # get the max DateTimeIndex and subtract 1 day
    max_date = df_train.index.max()
    logger.info(f"Max date: {max_date}")
    # training set
    features = features_config[target_var][:5]
    x_train = df_train.loc[: '2019-12-30', features]
    y_train = df_train.loc[: '2019-12-30', target_var]
    logger.info(f"Training data shape: {x_train.shape}")
    logger.info(f"Training data shape: {y_train.shape}")
    x_pred = df_train.loc['2019-12-31': '2020-01-01', features]
    y_actual = df_train.loc['2019-12-31': '2020-01-01', target_var]
    logger.info(f"Prediction shape: {x_pred.shape}")
    # create the gaussian ensemble model
    gpr_ensemble = GaussianProcessEnsemble()
    gpr_ensemble.train_kernels(x=x_train, y=y_train)

    y_pred, std_val = gpr_ensemble.predict_val(x=x_pred, return_std=False)
    logger.info(f"Prediction: {y_pred}\nActual: {y_actual}")

"""
Gaussian Process regression models for bond yield forecasting.
Implements different kernel types and model selection strategies.
"""

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct, ExpSineSquared, RBF, WhiteKernel, 
    ConstantKernel, Matern, RationalQuadratic
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_loader import BondDataLoader
from feature_manager import FeatureManager

import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianProcessEnsemble:
    """Ensemble of Gaussian Process models with different kernels."""
    
    def __init__(self, selection_metric: str = 'train_cosine_distance', random_state: int = 42, n_jobs: int = 3):
        """
        Initialize the GP ensemble.
        
        Args:
            random_state: Random state for reproducibility
        """
        metrics = ['train_cosine_distance', 'train_euclidean_rmse', 'train_r2_avg', 'train_r2_flat']
        if selection_metric not in metrics:
            raise ValueError(f"Unknown metric: {selection_metric}. Available metrics: {metrics}")
        self.selection_metric = selection_metric
        self.random_state = random_state
        self.kernels = None
        self.fitted_models: Dict[str, MultiOutputRegressor] = {}
        self.kernel_scores: Dict[str, Dict[str, float]] = {}
        self.best_kernel_name: Optional[str] = None
        self.n_jobs = n_jobs
        self.best_model: Optional[MultiOutputRegressor] = None
        self._create_kernel_configurations()


    def _create_kernel_configurations(self) -> None:
        """
        Create different kernel configurations for testing.
        
        Returns:
            Dictionary mapping kernel names to kernel objects
        """
        kernels = {
            # 'DotProduct': DotProduct() + WhiteKernel(noise_level=1e-5),
            
            # 'ExpSineSquared': (ConstantKernel(1.0, (1e-3, 1e3)) *
            #                   ExpSineSquared(length_scale=1.0, periodicity=1.0,
            #                                length_scale_bounds=(1e-2, 1e2),
            #                                periodicity_bounds=(1e-2, 1e2)) +
            #                   WhiteKernel(noise_level=1e-5)),

            # 'RBF': (ConstantKernel(1.0, (1e-3, 1e3)) *
            #        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
            #        WhiteKernel(noise_level=1e-5)),

            # 'WhiteKernel': WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1)),

            # 'Matern': (ConstantKernel(1.0, (1e-3, 1e3)) *
            #           Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
            #           WhiteKernel(noise_level=1e-5)),

            'RationalQuadratic': (ConstantKernel(1.0, (1e-3, 1e3)) *
                                 RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                 length_scale_bounds=(1e-2, 1e2),
                                                 alpha_bounds=(1e-5, 1e5)) +
                                 WhiteKernel(noise_level=1e-5)),

            # 'QuasiPeriodic': (ConstantKernel(1.0, (1e-3, 1e3)) *
            #                  RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) *
            #                  ExpSineSquared(length_scale=1.0, periodicity=1.0,
            #                               length_scale_bounds=(1e-2, 1e2),
            #                               periodicity_bounds=(1e-2, 1e2)) +
            #                  WhiteKernel(noise_level=1e-5))
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

    def _cosine_distance_avg(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine distance between two vectors.
        """
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


    def ensemble_train_runner(self, task: tuple[pd.DataFrame, pd.Series, str, int]) -> Dict[str, Any]:
        """
        task: tuple
            x: feature matrix
            y: target vector
            kernel_name: name of the kernel
        Returns:
            GaussianProcessRegressor
        """
        x, y, kernel_name = task
        # create a copy of the data
        x_copy = x.copy()
        y_copy = y.copy()
        model = self.create_gp_model(kernel_name=kernel_name)
        model = MultiOutputRegressor(model, n_jobs=3)

        # Fit model to get additional metrics
        model.fit(x_copy, y_copy)
        y_pred = model.predict(x_copy)
        # get the residuals
        residuals = y_copy.to_numpy() - y_pred

        estimators = model.estimators_
        metrics = {
            'kernel_name': kernel_name,
            'train_cosine_distance': self._cosine_distance_avg(y.to_numpy(), y_pred),
            'train_euclidean_rmse': self._euclidean_rmse_avg(y.to_numpy(), y_pred),
            'train_r2_avg': self.rsquared_score_avg(y.to_numpy(), y_pred),
            'train_r2_flat': self.rsquared_flat(y.to_numpy(), y_pred),
            'log_marginal_likelihood': [estimator.log_marginal_likelihood() for estimator in estimators],
            'residuals': residuals,
            'model': model,
        }
        return metrics


    def train_historical(self, x: pd.DataFrame, y: pd.Series):
        """
        Train all the Gaussian Process models using the different kernels concurrently.
        Returns:
        """
        target_std = np.std(y.to_numpy(), axis=0)
        tasks = [(x, y, kernel_name) for kernel_name in self.kernels.keys()]
        results = []

        # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        #     futures = [executor.submit(self.ensemble_train_runner,task) for task in tasks]
        #     for future in tqdm(as_completed(futures), desc='Kernels trained:', total=len(futures)):
        #         res = future.result()
        #         results.append(res)
        task = self.ensemble_train_runner(tasks[0])
        results.append(task)
        # return the best model
        results.sort(key=lambda x: x['train_cosine_distance'], reverse=True)
        self.kernel_scores = {x['kernel_name']: {
            'train_cosine_distance': x['train_cosine_distance'],
            'train_euclidean_rmse': x['train_euclidean_rmse'],
            'train_r2_avg': x['train_r2_avg'],
            'train_r2_flat': x['train_r2_flat'],
            'log_marginal_likelihood': x['log_marginal_likelihood'],
            'target_std': target_std,
            'residuals': x['residuals'],
            'model': x['model']
        } for x in results}
        # select the best model
        self._select_best_kernel()
        logger.info(f"Best kernel: {self.best_kernel_name}")

    
    def _select_best_kernel(self) -> None:
        """
        Select the best kernel based on cross-validation performance.
        metric: str
            Metric to use for selection. Options: 'train_cosine_distance', 'train_euclidean_rmse', 'train_r2_avg', 'train_r2_flat'
        Returns:
            None
        """
        # validate metric
        # logger.info(f"Evaluating kernel performance based on {self.selection_metric}...")
        if self.selection_metric == 'train_cosine_distance':
            self.best_kernel_name = max(self.kernel_scores, key=lambda x: self.kernel_scores[x]['train_cosine_distance'])
        elif self.selection_metric == 'train_euclidean_rmse':
            self.best_kernel_name = min(self.kernel_scores, key=lambda x: self.kernel_scores[x]['train_euclidean_rmse'])
        elif self.selection_metric == 'train_r2_avg':
            self.best_kernel_name = max(self.kernel_scores, key=lambda x: self.kernel_scores[x]['train_r2_avg'])
        else:
            self.best_kernel_name = max(self.kernel_scores, key=lambda x: self.kernel_scores[x]['train_r2_flat'])

        self.best_model = self.kernel_scores[self.best_kernel_name]['model']

    
    def predict_val(self, x: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Feature matrix for prediction
            return_std: Whether to return standard deviations
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        # self._select_best_kernel()
        

        return self.best_model.predict(x)




    def predict_val_distribution(self, x: pd.DataFrame, y: pd.Series, n_samples: int = 1000) -> np.ndarray:
        """
        Generate samples from the predictive distribution.

        Args:
            x: Feature matrix for prediction
            n_samples: Number of samples to draw
        """
        if self.best_model is None:
            raise ValueError("Must fit best model first")
        residuals = self.kernel_scores[self.best_kernel_name]['residuals']
        # get the mean prediction
        y_mean = self.best_model.predict(x)
        y_samples = np.array([
            np.random.choice(residuals[:, col], size=n_samples, replace=True) for col in range(residuals.shape[1])
                        ]).T
        y_samples = y_mean + y_samples
        return y_samples.T[np.newaxis, :, :]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the fitted models.
        
        Returns:
            Dictionary with model summary information
        """
        if self.best_model is None:
            return {"status": "No model fitted"}

        models = self.best_model.estimators_
        summary = {
            "best_kernel": self.best_kernel_name,
            "best_kernel_params": [str(model.kernel_) for model in models],
            "log_marginal_likelihood": [model.log_marginal_likelihood() for model in models],
            # "kernel_scores": self.kernel_scores,
            # "available_kernels": list(self.kernels.keys()),
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
        model_list = self.best_model.estimators_
        importance_scores_list = []
        for model in model_list:
            if hasattr(model.kernel_, 'length_scale'):
                length_scales = self.best_model.kernel_.length_scale

                if hasattr(length_scales, '__len__') and len(length_scales) == X.shape[1]:
                    # Invert length scales: smaller length scale = higher importance
                    importance_scores = 1.0 / (length_scales + 1e-10)
                    importance_scores = importance_scores / importance_scores.sum()

                    importance_scores_list.append(pd.Series(importance_scores, index=X.columns, name='importance'))

            # Fallback: uniform importance
            logger.warning("Could not extract feature importance, using uniform weights")
            uniform_importance = np.ones(X.shape[1]) / X.shape[1]
            importance_scores_list.append(pd.Series(uniform_importance, index=X.columns, name='importance'))
        return pd.concat(importance_scores_list)


if __name__ == "__main__":
    # load the data
    # df_train = pd.read_parquet("/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/bond_yields_ns_params_shifted_1.parquet")
    data_loader = BondDataLoader(data_path="/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/bond_yields_ns_params_shifted_1.parquet")
    # load the feature config yaml file
    feature_manager = FeatureManager(features_config_path="/Users/mma0277/Documents/Development/investment_analysis/tt-investment-analysis/data/project_work/features_selected.yaml")
    features = feature_manager.features_config['one-day-ahead']['max_features']
    target_var = feature_manager.get_dependent_variables()
    data_loader.load_data(x=features, y=target_var)
    logger.info(f"Loaded dataset with {len(data_loader.data)} rows")
    # get the max DateTimeIndex and subtract 1 day
    max_date = data_loader.data.index.max()
    logger.info(f"Max date: {max_date}")
    windows = data_loader.get_time_windows(window_size=252, min_window_size=100)
    # training set
    x_train, y_train = data_loader.get_window_data(start_idx=windows[-2][0], end_idx=windows[-2][1],
                                         target_columns=target_var, feature_columns=features)
    logger.info(f"Training data shape: X={x_train.shape}, y={y_train.shape}")
    logger.info(f"Training data: {x_train.index.max()}")
    # get prediction point (the last point in the data)
    pred_features = data_loader.get_prediction_point(idx=windows[-2][1], feature_columns=features)
    actual_value = data_loader.get_actual_value(idx=windows[-2][1], target_columns=target_var)
    logger.info(f"Prediction features: {pred_features.index.max()}")
    print(f"Actual target value: {actual_value.to_numpy()}")
    pred_date = data_loader.get_date_for_index(idx=windows[-2][1])
    # create the gaussian ensemble model
    gpr_ensemble = GaussianProcessEnsemble()
    gpr_ensemble.train_historical(x=x_train, y=y_train)

    y_pred, std_val = gpr_ensemble.predict_val(x=pred_features, return_std=False)
    y_samples = gpr_ensemble.predict_val_distribution(x=pred_features, n_samples=1000)
    logger.info(f"Prediction: {y_pred}\nActual: {actual_value.to_numpy()}")
    logger.info(f"Y samples shape: {y_samples.shape}")
    samples_df = pd.DataFrame(y_samples.reshape(-1, y_samples.shape[-1]).T,
                              columns=target_var)
    print(samples_df.head(10))
    model_summary = gpr_ensemble.get_model_summary()
    logger.info(f"Model Summary: {model_summary}")
    feature_importance = gpr_ensemble.get_feature_importance_proxy(X=x_train)
    logger.info(f"Feature Importance Proxy:\n{feature_importance.sort_values(ascending=False)}")

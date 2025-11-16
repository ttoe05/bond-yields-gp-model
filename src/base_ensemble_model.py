"""
Abstract base class for ensemble models in bond yield forecasting pipeline.
Provides standardized interface across GP, Bayesian Ridge, and Kernel Ridge models.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from random import uniform
import logging

logger = logging.getLogger(__name__)


class BaseEnsembleModel(ABC):
    """Abstract base class for all ensemble models in the forecasting pipeline."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize base ensemble model.
        
        Args:
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs for training
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.is_trained = False
        self.training_columns: Optional[List[str]] = None
        self.training_percentiles: Optional[Dict[str, tuple]] = None
        
    @abstractmethod
    def train_historical(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the model on historical data.
        
        Args:
            x: Feature DataFrame
            y: Target DataFrame
            
        Returns:
            Dictionary with training metrics and model information
        """
        pass
    
    @abstractmethod
    def predict_val(self, x: pd.DataFrame) -> np.ndarray:
        """
        Make point predictions.
        
        Args:
            x: Feature DataFrame for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_val_distribution(self, x: pd.DataFrame, y: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate prediction distribution samples.
        
        Args:
            x: Feature DataFrame for prediction
            y: Target DataFrame (for scaling/reference)
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with prediction samples
        """
        pass
    
    @abstractmethod
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of trained model performance and parameters.
        
        Returns:
            Dictionary with model summary information
        """
        pass
    
    @abstractmethod
    def get_feature_importance_proxy(self, X: pd.DataFrame) -> pd.Series:
        """
        Get feature importance or proxy measure.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Series with feature importance scores
        """
        pass
    
    def _validate_inputs(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """
        Validate input data.
        
        Args:
            x: Feature DataFrame
            y: Optional target DataFrame
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Features must be provided as pandas DataFrame")
        
        if x.empty:
            raise ValueError("Feature DataFrame cannot be empty")
        
        if x.isnull().any().any():
            raise ValueError("Feature DataFrame contains null values")
        
        if y is not None:
            if not isinstance(y, pd.DataFrame):
                raise ValueError("Targets must be provided as pandas DataFrame")
            
            if y.empty:
                raise ValueError("Target DataFrame cannot be empty")
            
            if len(x) != len(y):
                raise ValueError("Feature and target DataFrames must have same length")
    
    def _validate_trained(self) -> None:
        """
        Validate that model has been trained.
        
        Raises:
            ValueError: If model has not been trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
    
    def _apply_prediction_boundaries(self, predictions: np.ndarray, column_names: List[str]) -> np.ndarray:
        """
        Apply prediction boundaries to prevent extreme outliers and negative yields.
        
        Args:
            predictions: Raw prediction array
            column_names: Names of predicted columns
            
        Returns:
            Bounded prediction array
        """
        if self.training_percentiles is None or self.training_columns is None:
            logger.warning("Training boundaries not available, skipping boundary enforcement")
            return predictions
        
        bounded_predictions = predictions.copy()
        # calcluate the 3rd and 4th value from the standard deviation
        for i, col in enumerate(column_names):
            if col in self.training_percentiles:
                # td_3 = self.training_percentiles[col][3]
                p1, p99, mean_val, std_val = self.training_percentiles[col]
                clip_val1 = mean_val + (3.5 * std_val)
                clip_val2 = mean_val + (4.6 * std_val)
                # randomly sample a number between these two value
                clip_val = uniform(clip_val1, clip_val2)
                
                # Apply percentile boundaries
                if predictions.ndim == 1:
                    bounded_predictions[i] = np.clip(bounded_predictions[i], p1, clip_val)
                else:
                    bounded_predictions[:, i] = np.clip(bounded_predictions[:, i], p1, clip_val)
                
                # Apply yield floor (bond yields should not be negative)
                if predictions.ndim == 1:
                    bounded_predictions[i] = max(bounded_predictions[i], uniform(0.001, 0.1))  # 0.01%
                else:
                    bounded_predictions[:, i] = np.maximum(bounded_predictions[:, i], uniform(0.001, 0.1))
        
        return bounded_predictions
    
    def _store_training_boundaries(self, y: pd.DataFrame) -> None:
        """
        Calculate and store training data boundaries for prediction enforcement.
        
        Args:
            y: Training target DataFrame
        """
        self.training_columns = list(y.columns)
        self.training_percentiles = {}
        
        for col in self.training_columns:
            p1 = np.percentile(y[col].dropna(), 1)
            p99 = np.percentile(y[col].dropna(), 99)
            mean_val = y[col].mean()
            std_val = y[col].std()
            self.training_percentiles[col] = (p1, p99, mean_val, std_val)
        
        logger.debug(f"Stored training boundaries for {len(self.training_columns)} target columns")
    
    def apply_yield_constraints(self, predictions: Union[np.ndarray, pd.DataFrame], 
                              is_inverse_transformed: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply yield constraints to ensure non-negative yields.
        
        Args:
            predictions: Prediction array or DataFrame
            is_inverse_transformed: Whether predictions have been inverse transformed from scaling
            
        Returns:
            Constrained predictions in same format as input
        """
        if isinstance(predictions, pd.DataFrame):
            constrained = predictions.copy()
            # Apply minimum yield constraint (0.01%)
            constrained = constrained.clip(lower=uniform(0.001, 0.01))
            return constrained
        else:
            # Apply minimum yield constraint (0.01%)
            return np.maximum(predictions, uniform(0.001, 0.01))
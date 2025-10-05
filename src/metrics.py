"""
Comprehensive metrics calculation for bond yield forecasting evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingMetrics:
    """Comprehensive forecasting metrics calculator."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_calculated: Dict[str, Any] = {}
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic forecasting metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with basic metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions for basic metrics")
            return self._get_invalid_metrics()
        
        metrics = {
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'mse': mean_squared_error(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'r2': r2_score(y_true_clean, y_pred_clean),
            'n_samples': len(y_true_clean)
        }
        
        return metrics
    
    def calculate_percentage_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate percentage-based metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with percentage metrics
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true != 0)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions for percentage metrics")
            return {'mape': np.nan, 'smape': np.nan}
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(y_pred_clean - y_true_clean) / 
                       (np.abs(y_pred_clean) + np.abs(y_true_clean))) * 100
        
        return {
            'mape': mape,
            'smape': smape
        }
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with directional metrics
        """
        if len(y_true) < 2:
            return {'directional_accuracy': np.nan, 'up_accuracy': np.nan, 'down_accuracy': np.nan}
        
        # Calculate changes
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Remove NaN changes
        mask = ~(np.isnan(true_changes) | np.isnan(pred_changes))
        true_changes = true_changes[mask]
        pred_changes = pred_changes[mask]
        
        if len(true_changes) == 0:
            return {'directional_accuracy': np.nan, 'up_accuracy': np.nan, 'down_accuracy': np.nan}
        
        # Direction indicators
        true_up = true_changes > 0
        pred_up = pred_changes > 0
        
        # Overall directional accuracy
        directional_accuracy = np.mean(true_up == pred_up) * 100
        
        # Up movement accuracy
        up_movements = np.sum(true_up)
        up_accuracy = np.mean(pred_up[true_up]) * 100 if up_movements > 0 else np.nan
        
        # Down movement accuracy
        down_movements = np.sum(~true_up)
        down_accuracy = np.mean(~pred_up[~true_up]) * 100 if down_movements > 0 else np.nan
        
        return {
            'directional_accuracy': directional_accuracy,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'n_up_movements': up_movements,
            'n_down_movements': down_movements
        }
    
    def calculate_distribution_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate distribution comparison metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with distribution metrics
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 3:
            return self._get_invalid_distribution_metrics()
        
        try:
            # Kolmogorov-Smirnov test
            ks_statistic, ks_pvalue = stats.ks_2samp(y_true_clean, y_pred_clean)
            
            # Mean and variance comparison
            true_mean, true_std = np.mean(y_true_clean), np.std(y_true_clean)
            pred_mean, pred_std = np.mean(y_pred_clean), np.std(y_pred_clean)
            
            # Bias metrics
            bias = pred_mean - true_mean
            relative_bias = bias / true_mean if true_mean != 0 else np.nan
            
            return {
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue,
                'bias': bias,
                'relative_bias': relative_bias,
                'true_mean': true_mean,
                'pred_mean': pred_mean,
                'true_std': true_std,
                'pred_std': pred_std,
                'std_ratio': pred_std / true_std if true_std != 0 else np.nan
            }
            
        except Exception as e:
            logger.warning(f"Error calculating distribution metrics: {str(e)}")
            return self._get_invalid_distribution_metrics()
    
    def calculate_uncertainty_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate uncertainty calibration metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_std: Prediction standard deviations
            
        Returns:
            Dictionary with uncertainty metrics
        """
        if y_std is None:
            return {'coverage_95': np.nan, 'coverage_68': np.nan, 'avg_interval_width': np.nan}
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_std))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        y_std_clean = y_std[mask]
        
        if len(y_true_clean) == 0:
            return {'coverage_95': np.nan, 'coverage_68': np.nan, 'avg_interval_width': np.nan}
        
        # Calculate confidence intervals
        ci_95_lower = y_pred_clean - 1.96 * y_std_clean
        ci_95_upper = y_pred_clean + 1.96 * y_std_clean
        ci_68_lower = y_pred_clean - 1.0 * y_std_clean
        ci_68_upper = y_pred_clean + 1.0 * y_std_clean
        
        # Coverage probabilities
        coverage_95 = np.mean((y_true_clean >= ci_95_lower) & (y_true_clean <= ci_95_upper)) * 100
        coverage_68 = np.mean((y_true_clean >= ci_68_lower) & (y_true_clean <= ci_68_upper)) * 100
        
        # Average interval width
        avg_interval_width_95 = np.mean(ci_95_upper - ci_95_lower)
        avg_interval_width_68 = np.mean(ci_68_upper - ci_68_lower)
        
        return {
            'coverage_95': coverage_95,
            'coverage_68': coverage_68,
            'avg_interval_width_95': avg_interval_width_95,
            'avg_interval_width_68': avg_interval_width_68,
            'avg_prediction_std': np.mean(y_std_clean)
        }
    
    def calculate_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Perform residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residual analysis results
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 3:
            return self._get_invalid_residual_metrics()
        
        residuals = y_true_clean - y_pred_clean
        
        try:
            # Normality test
            shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
            
            # Autocorrelation (lag-1)
            if len(residuals) > 1:
                autocorr_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            else:
                autocorr_lag1 = np.nan
            
            # Heteroscedasticity test (simple)
            # Split residuals into first and second half and compare variances
            mid_point = len(residuals) // 2
            first_half_var = np.var(residuals[:mid_point])
            second_half_var = np.var(residuals[mid_point:])
            
            variance_ratio = second_half_var / first_half_var if first_half_var != 0 else np.nan
            
            return {
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': stats.skew(residuals),
                'residual_kurtosis': stats.kurtosis(residuals),
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_pvalue,
                'autocorr_lag1': autocorr_lag1,
                'variance_ratio': variance_ratio,
                'residual_min': np.min(residuals),
                'residual_max': np.max(residuals)
            }
            
        except Exception as e:
            logger.warning(f"Error in residual analysis: {str(e)}")
            return self._get_invalid_residual_metrics()
    
    def calculate_all_metrics(self, y_true: Union[np.ndarray, pd.Series], 
                            y_pred: Union[np.ndarray, pd.Series],
                            y_std: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_std: Prediction standard deviations (optional)
            
        Returns:
            Dictionary with all metrics
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        if isinstance(y_std, pd.Series):
            y_std = y_std.values
        
        logger.info("Calculating comprehensive forecasting metrics")
        
        all_metrics = {
            'basic_metrics': self.calculate_basic_metrics(y_true, y_pred),
            'percentage_metrics': self.calculate_percentage_metrics(y_true, y_pred),
            'directional_metrics': self.calculate_directional_accuracy(y_true, y_pred),
            'distribution_metrics': self.calculate_distribution_metrics(y_true, y_pred),
            'uncertainty_metrics': self.calculate_uncertainty_metrics(y_true, y_pred, y_std),
            'residual_analysis': self.calculate_residual_analysis(y_true, y_pred)
        }
        
        # Store for later access
        self.metrics_calculated = all_metrics
        
        return all_metrics
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of key metrics.
        
        Returns:
            DataFrame with key metrics summary
        """
        if not self.metrics_calculated:
            return pd.DataFrame()
        
        key_metrics = {
            'MAE': self.metrics_calculated['basic_metrics'].get('mae'),
            'RMSE': self.metrics_calculated['basic_metrics'].get('rmse'),
            'R²': self.metrics_calculated['basic_metrics'].get('r2'),
            'MAPE (%)': self.metrics_calculated['percentage_metrics'].get('mape'),
            'Directional Accuracy (%)': self.metrics_calculated['directional_metrics'].get('directional_accuracy'),
            'Coverage 95% (%)': self.metrics_calculated['uncertainty_metrics'].get('coverage_95'),
            'Bias': self.metrics_calculated['distribution_metrics'].get('bias')
        }
        
        summary_df = pd.DataFrame(list(key_metrics.items()), 
                                columns=['Metric', 'Value'])
        
        return summary_df
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export all metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        if not self.metrics_calculated:
            logger.warning("No metrics calculated to export")
            return
        
        try:
            import json
            
            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                return obj
            
            metrics_json = convert_numpy_types(self.metrics_calculated)
            
            with open(filepath, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            raise
    
    def _get_invalid_metrics(self) -> Dict[str, float]:
        """Get invalid metrics template."""
        return {
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'n_samples': 0
        }
    
    def _get_invalid_distribution_metrics(self) -> Dict[str, float]:
        """Get invalid distribution metrics template."""
        return {
            'ks_statistic': np.nan,
            'ks_pvalue': np.nan,
            'bias': np.nan,
            'relative_bias': np.nan,
            'true_mean': np.nan,
            'pred_mean': np.nan,
            'true_std': np.nan,
            'pred_std': np.nan,
            'std_ratio': np.nan
        }
    
    def _get_invalid_residual_metrics(self) -> Dict[str, float]:
        """Get invalid residual metrics template."""
        return {
            'residual_mean': np.nan,
            'residual_std': np.nan,
            'residual_skewness': np.nan,
            'residual_kurtosis': np.nan,
            'shapiro_statistic': np.nan,
            'shapiro_pvalue': np.nan,
            'autocorr_lag1': np.nan,
            'variance_ratio': np.nan,
            'residual_min': np.nan,
            'residual_max': np.nan
        }
    
    @staticmethod
    def compare_models(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models' metrics.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame comparing models
        """
        comparison_data = {}
        
        for model_name, metrics in metrics_dict.items():
            basic_metrics = metrics.get('basic_metrics', {})
            percentage_metrics = metrics.get('percentage_metrics', {})
            directional_metrics = metrics.get('directional_metrics', {})
            uncertainty_metrics = metrics.get('uncertainty_metrics', {})
            
            comparison_data[model_name] = {
                'MAE': basic_metrics.get('mae'),
                'RMSE': basic_metrics.get('rmse'),
                'R²': basic_metrics.get('r2'),
                'MAPE': percentage_metrics.get('mape'),
                'Directional_Acc': directional_metrics.get('directional_accuracy'),
                'Coverage_95': uncertainty_metrics.get('coverage_95')
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        return comparison_df
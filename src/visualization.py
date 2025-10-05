"""
Visualization utilities for bond yield forecasting results.

This module provides functions to create various plots and visualizations
for analyzing the performance of Gaussian Process regression models
on bond yield forecasting tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class ForecastVisualizer:
    """Class for creating various visualizations of forecasting results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer with plotting preferences.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size for plots
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_forecast_results(self, 
                            dates: pd.DatetimeIndex,
                            actual: np.ndarray,
                            predictions: np.ndarray,
                            confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                            title: str = "Bond Yield Forecast Results",
                            bond_name: str = "DGS10") -> plt.Figure:
        """
        Plot actual vs predicted values with confidence intervals.
        
        Args:
            dates: Time index for the data
            actual: Actual values
            predictions: Predicted values
            confidence_intervals: Tuple of (lower_bound, upper_bound) arrays
            title: Plot title
            bond_name: Name of the bond index
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual values
        ax.plot(dates, actual, label='Actual', color=self.colors[0], linewidth=2)
        
        # Plot predictions
        ax.plot(dates, predictions, label='Predicted', color=self.colors[1], 
                linewidth=2, alpha=0.8)
        
        # Add confidence intervals if provided
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            ax.fill_between(dates, lower, upper, alpha=0.3, color=self.colors[1],
                          label='95% Confidence Interval')
        
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{bond_name} Yield (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_residuals(self, 
                      actual: np.ndarray,
                      predictions: np.ndarray,
                      title: str = "Residual Analysis") -> plt.Figure:
        """
        Create residual plots for model diagnostics.
        
        Args:
            actual: Actual values
            predictions: Predicted values
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        residuals = actual - predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Residuals vs Fitted
        axes[0, 0].scatter(predictions, residuals, alpha=0.6, color=self.colors[0])
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # QQ plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, density=True, alpha=0.7, color=self.colors[2])
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals over time
        axes[1, 1].plot(residuals, color=self.colors[3])
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, 
                              metrics_dict: Dict[str, Dict[str, float]],
                              title: str = "Model Performance Comparison") -> plt.Figure:
        """
        Compare performance metrics across different models or time periods.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metrics as values
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        df = pd.DataFrame(metrics_dict).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        metrics = ['MAE', 'RMSE', 'MAPE', 'Directional_Accuracy']
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=axes[row, col], color=self.colors[i])
                axes[row, col].set_title(f'{metric}')
                axes[row, col].set_ylabel(metric)
                axes[row, col].tick_params(axis='x', rotation=45)
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_kernel_performance(self, 
                              kernel_results: Dict[str, float],
                              title: str = "Kernel Performance Comparison") -> plt.Figure:
        """
        Visualize performance of different GP kernels.
        
        Args:
            kernel_results: Dictionary with kernel names as keys and scores as values
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        kernels = list(kernel_results.keys())
        scores = list(kernel_results.values())
        
        # Bar plot
        bars = ax1.bar(kernels, scores, color=self.colors[:len(kernels)])
        ax1.set_title('Kernel Performance (Higher is Better)')
        ax1.set_ylabel('Cross-Validation Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(scores, labels=kernels, autopct='%1.1f%%', colors=self.colors[:len(kernels)])
        ax2.set_title('Kernel Performance Distribution')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_forecast_plot(self, 
                                       dates: pd.DatetimeIndex,
                                       actual: np.ndarray,
                                       predictions: np.ndarray,
                                       confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                       bond_name: str = "DGS10") -> go.Figure:
        """
        Create an interactive Plotly visualization of forecast results.
        
        Args:
            dates: Time index for the data
            actual: Actual values
            predictions: Predicted values
            confidence_intervals: Tuple of (lower_bound, upper_bound) arrays
            bond_name: Name of the bond index
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence intervals if provided
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            
            # Upper bound
            fig.add_trace(go.Scatter(
                x=dates,
                y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=dates,
                y=lower,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.3)',
                fill='tonexty',
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f'{bond_name} Yield Forecast Results',
            xaxis_title='Date',
            yaxis_title=f'{bond_name} Yield (%)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_rolling_metrics(self, 
                           dates: pd.DatetimeIndex,
                           metrics_series: pd.Series,
                           metric_name: str,
                           window_size: int = 30,
                           title: str = None) -> plt.Figure:
        """
        Plot rolling metrics over time.
        
        Args:
            dates: Time index
            metrics_series: Series of metric values
            metric_name: Name of the metric
            window_size: Rolling window size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        if title is None:
            title = f"Rolling {metric_name} (Window: {window_size} days)"
        
        rolling_metric = metrics_series.rolling(window=window_size).mean()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates, rolling_metric, label=f'Rolling {metric_name}', 
                color=self.colors[0], linewidth=2)
        ax.fill_between(dates, rolling_metric, alpha=0.3, color=self.colors[0])
        
        ax.set_xlabel('Date')
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def save_all_plots(self, 
                      save_dir: str,
                      dates: pd.DatetimeIndex,
                      actual: np.ndarray,
                      predictions: np.ndarray,
                      confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      metrics_dict: Optional[Dict[str, float]] = None,
                      kernel_results: Optional[Dict[str, float]] = None,
                      bond_name: str = "DGS10"):
        """
        Save all standard plots to a directory.
        
        Args:
            save_dir: Directory to save plots
            dates: Time index
            actual: Actual values
            predictions: Predicted values
            confidence_intervals: Tuple of confidence bounds
            metrics_dict: Dictionary of performance metrics
            kernel_results: Dictionary of kernel performance results
            bond_name: Name of the bond index
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Main forecast plot
        fig1 = self.plot_forecast_results(dates, actual, predictions, 
                                         confidence_intervals, bond_name=bond_name)
        fig1.savefig(f"{save_dir}/{bond_name}_forecast_results.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Residual analysis
        fig2 = self.plot_residuals(actual, predictions)
        fig2.savefig(f"{save_dir}/{bond_name}_residual_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Metrics comparison if provided
        if metrics_dict is not None:
            fig3 = self.plot_metrics_comparison({bond_name: metrics_dict})
            fig3.savefig(f"{save_dir}/{bond_name}_metrics_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        # Kernel performance if provided
        if kernel_results is not None:
            fig4 = self.plot_kernel_performance(kernel_results)
            fig4.savefig(f"{save_dir}/{bond_name}_kernel_performance.png", dpi=300, bbox_inches='tight')
            plt.close(fig4)
        
        print(f"All plots saved to {save_dir}")


def create_summary_report(results_dict: Dict, save_path: str = None) -> pd.DataFrame:
    """
    Create a summary report of forecasting results.
    
    Args:
        results_dict: Dictionary containing all results
        save_path: Path to save the report CSV
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for bond_name, results in results_dict.items():
        summary_data.append({
            'Bond_Index': bond_name,
            'MAE': results.get('MAE', np.nan),
            'RMSE': results.get('RMSE', np.nan),
            'MAPE': results.get('MAPE', np.nan),
            'Directional_Accuracy': results.get('Directional_Accuracy', np.nan),
            'Best_Kernel': results.get('Best_Kernel', 'Unknown'),
            'Training_Time': results.get('Training_Time', np.nan),
            'Predictions_Count': results.get('Predictions_Count', np.nan)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"Summary report saved to {save_path}")
    
    return summary_df
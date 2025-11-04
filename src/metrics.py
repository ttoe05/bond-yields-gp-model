"""
Comprehensive metrics calculation for bond yield forecasting evaluation.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from seaborn import color_palette
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dgs_percentile_calc import get_list_sample_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingMetrics:
    """Comprehensive forecasting metrics calculator."""
    
    def __init__(self, time_prediction: str, actuals_file: str) -> None:
        """Initialize metrics calculator."""
        self.time_prediction = time_prediction
        self.actuals_df: pd.DataFrame = None
        self.forecasts_df: pd.DataFrame = None
        self.actuals_file = actuals_file
        self.dependent_varaibles = [
            'DGS1MO_future_val', 'DGS3MO_future_val', 'DGS6MO_future_val',
            'DGS1_future_val', 'DGS2_future_val', 'DGS3_future_val', 'DGS5_future_val',
            'DGS7_future_val', 'DGS10_future_val', 'DGS20_future_val', 'DGS30_future_val'
        ]
        self.actuals_variables = [
            'DGS1MO', 'DGS3MO', 'DGS6MO',
            'DGS1', 'DGS2', 'DGS3', 'DGS5',
            'DGS7', 'DGS10', 'DGS20', 'DGS30'
        ]
        self._get_actuals_df()
        logger.info(self.actuals_df.head())
        self._get_forecast_df()

    def _get_actuals_df(self) -> None:
        """Get the file path for actuals based on time prediction."""
        # generate case-specific file path
        # read in the actuals data
        df = pd.read_parquet(self.actuals_file)
        df = df[df.index.to_timestamp() >= pd.to_datetime('2010-01-01')]
        match self.time_prediction:
            case 'one-day-ahead':
                self.actuals_df = df[self.actuals_variables].shift(-1)
            case 'seven-day-ahead':
                self.actuals_df = df[self.actuals_variables].shift(-7)
            case 'thirty-day-ahead':
                self.actuals_df = df[self.actuals_variables].shift(-30)
            case 'sixty-day-ahead':
                self.actuals_df = df[self.actuals_variables].shift(-60)
            case _:
                raise ValueError(f"Invalid time_prediction: {self.time_prediction}")

    def _get_forecast_record(self, file: str) -> dict[str, Any]:
        """
        Get the forecast record from the given file.
        Args:
            file:
        Returns: Dict with forecast record
        """
        # Read the data from the file
        df_tmp = pd.read_parquet(file)
        record = {}
        # get the date from the file name and convert to datetime
        date_str = file.split('/')[-1].replace('.parquet', '')
        record['date'] = pd.to_datetime(date_str)
        for col in self.dependent_varaibles:
            record[f"{col}_mean"] = df_tmp[col].mean()
            record[f"{col}_97_5"] = df_tmp[df_tmp[f"{col}_percentile"] == 0.975][col].values[0]
            record[f"{col}_2_5"] = df_tmp[df_tmp[f"{col}_percentile"] == 0.025][col].values[0]

        return record

    def _get_forecast_df(self)-> None:
        """
        Get the forecast Dataframe with the 95% confidence bands
        Returns: None
        """
        # get the list of sample files
        sample_files = get_list_sample_files(time_prediction=self.time_prediction, percentile=True)
        max_workers = 8
        # user mutlithreading to read in all the forecast files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            records = list(tqdm(executor.map(self._get_forecast_record, sample_files),
                                total=len(sample_files)))

        self.forecasts_df = pd.DataFrame(records)
        # combine with actuals
        sample_df_tmp = self.actuals_df.copy()
        sample_df_tmp['date'] = sample_df_tmp.index.to_timestamp()
        self.forecasts_df = self.forecasts_df.merge(sample_df_tmp, left_on='date', right_on='date', how='left')


    def _create_confidence_plot_df(self, df: pd.DataFrame, col: str, confidence_bands: bool) -> pd.DataFrame:
        """
        Create a DataFrame for confidence plot for a given column.
        Args:
            df: DataFrame with actuals and forecasts
            col: Column name to create plot for
        """
        # remove appendix from col name if exists
        col_rename = col.replace('_future_val', '')
        if confidence_bands:
            full_cols = ['date', col_rename, f"{col}_mean", f"{col}_2_5", f"{col}_97_5"]
            value_vars = [col_rename, f"{col}_mean", f"{col}_2_5", f"{col}_97_5"]
        else:
            full_cols = ['date', col_rename, f"{col}_mean"]
            value_vars = [col_rename, f"{col}_mean"]
        df_tmp = df[full_cols].copy()
        return df_tmp.melt(id_vars=['date'],
                           value_vars=value_vars,
                           var_name='type',
                           value_name='value')

    def plot_time_confidence_intervals(self, train: bool, confidence_bands: bool, save_fig: bool) -> None:
        """
        Generate DataFrame for plotting confidence intervals for a given column.
        Args:
            col: Column name to plot

        """
        # create a 4 by 3 subplot of the confidence intervals for each bond index
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 17))
        axes = axes.flatten()
        for i, col in enumerate(self.dependent_varaibles):
            plot_df = self._create_confidence_plot_df(self.forecasts_df, col, confidence_bands)
            col_rename = col.replace('_future_val', '')
            if train:
                plot_df = plot_df[(plot_df['date'] < pd.to_datetime('2018-01-01')) & (plot_df['date'] > pd.to_datetime('2014-01-01'))]
            color_palette = {
                col_rename: 'blue',
                f"{col}_mean": 'orange',
                f"{col}_2_5": 'red',
                f"{col}_97_5": 'red'
            }
            sns.lineplot(plot_df, x='date', y='value', hue='type',
                         palette=color_palette, ax=axes[i])
            axes[i].set_title(f'Confidence Intervals for {col} - {self.time_prediction}')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Yield (%)')
            axes[i].legend()
        plt.tight_layout()
        if save_fig:
            # create plots directory if not exists
            Path(f'plots/{self.time_prediction}/').mkdir(parents=True, exist_ok=True)
            plt.savefig(f'plots/{self.time_prediction}/confidence_intervals_{"train" if train else "full"}.png')
        plt.show()

    def plot_actuals(self):
        """
        Plot actual bond yields over time.
        """
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 17))
        axes = axes.flatten()
        for i, col in enumerate(self.dependent_varaibles):
            col_rename = col.replace('_future_val', '')
            sns.lineplot(data=self.actuals_df, x=self.actuals_df.index.to_timestamp(), y=col_rename, ax=axes[i])
            axes[i].set_title(f'Actual Yields for {col_rename} - {self.time_prediction}')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Yield (%)')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    forecasts = ForecastingMetrics(time_prediction='thirty-day-ahead',
                                   actuals_file="data/fred_prorcessed_daily.parquet")

    print(forecasts.forecasts_df.head())
    print(forecasts.actuals_df.head())
    # print(forecasts._create_confidence_plot_df(forecasts.forecasts_df, 'DGS1MO').head())
    forecasts.plot_time_confidence_intervals(train=True, confidence_bands=True, save_fig=True)
    forecasts.plot_time_confidence_intervals(train=False, confidence_bands=False, save_fig=True)
    forecasts.plot_time_confidence_intervals(train=False, confidence_bands=True, save_fig=True)
    forecasts.plot_time_confidence_intervals(train=True, confidence_bands=False, save_fig=True)
    forecasts.plot_actuals()



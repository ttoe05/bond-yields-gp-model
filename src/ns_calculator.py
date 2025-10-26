"""
Transform all outputs, samples, point estimated values, and actual coefficients into the DGS
yield indexes
"""
import sys

import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
#import threadpool executor for parallel processing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from itertools import repeat
from pathlib import Path

MATURITIES = [round(1/12, 3), 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30]
DEPENDENT_COLUMN_NAMES = [
    'DGS1MO', 'DGS3MO', 'DGS6MO',
    'DGS1', 'DGS2', 'DGS3', 'DGS5',
    'DGS7', 'DGS10', 'DGS20', 'DGS30'
]

def get_list_of_sample_files(time_prediction: str, yields: bool = False) -> list[str]:
    """
    Get a list of all sample files for a given time prediction.
    Args:
        time_prediction: str
            The time prediction identifier (e.g., 'one-day-ahead', 'seven-day-ahead', etc.)
    Returns:
        list[str]: List of file paths to sample files.
    """
    # check the time_prediction passed in is valid
    if time_prediction not in ['one-day-ahead', 'seven-day-ahead', 'fourteen-day-ahead', 'thirty-day-ahead']:
        raise ValueError(f"Invalid time_prediction: {time_prediction}. Must be one of 'one-day-ahead', 'seven-day-ahead', 'fourteen-day-ahead', 'thirty-day-ahead'.")
    if yields:
        sample_dir = f"results/{time_prediction}/samples/yields/"
    else:
        sample_dir = f"results/{time_prediction}/samples/"
    sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.parquet')]
    return sample_files


def nelson_siegel(maturities: np.ndarray, beta0: float, beta1: float, beta2: float, lambda_: float) -> np.ndarray:
    """Nelson-Siegel yield curve model."""
    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-lambda_ * maturities)) / (lambda_ * maturities))
    term3 = beta2 * (((1 - np.exp(-lambda_ * maturities)) / (lambda_ * maturities)) - np.exp(-lambda_ * maturities))
    return term1 + term2 + term3


# create a function that reads in the generated samples and transforms them into yield indexes
def transform_sample_to_yields(sample_file: str, time_prediction: str) -> None:
    """
    Transform a single sample file of NS parameters into yield indexes.
    and persist the results to a new file.
    Args:
        sample_file: str
            Path to the sample file containing NS parameters.
        time_prediction: str
            The time prediction identifier (e.g., 'one-day-ahead', 'seven-day-a
    Returns: None
    """
    # read the parquet files
    samples_df = pd.read_parquet(sample_file)
    yield_values = samples_df.apply(lambda row: nelson_siegel(np.array(MATURITIES), *row.to_numpy()) * 100, axis=1)
    df_yield_distribution = pd.DataFrame(yield_values.tolist(), columns=DEPENDENT_COLUMN_NAMES)
    # create the percentile rank columns
    for col in DEPENDENT_COLUMN_NAMES:
        df_yield_distribution[f"{col}_percentile"] = df_yield_distribution[col].rank(pct=True)
    # persist the results
    df_yield_distribution.to_parquet(f"results/{time_prediction}/samples/yields/{sample_file.split('/')[-1]}")


def transform_all_samples_to_yields(sample_files: list[str], time_prediction: str, max_workers: int=4) -> None:
    """
    Concurrently transform all sample files of NS parameters into yield indexes.
    Args:
        sample_files:
        time_prediction:
        max_workers:
    Returns: None
    """
    # create the yields subdirectory if it doesn't exist
    yields_dir = Path(f'results/{time_prediction}/samples/yields/')
    Path(yields_dir).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(transform_sample_to_yields, sample_files, repeat(time_prediction)), total=len(sample_files)))


def transform_point_estimates_to_yields(time_prediction: str) -> None:
    """
    Transform point estimated NS parameters into yield indexes.
    Args:
        time_prediction: str
            The time prediction identifier (e.g., 'one-day-ahead', 'seven-day-ahead', etc.)
    Returns: None
    """
    point_estimate_file = f"results/{time_prediction}/{time_prediction}_results.parquet"
    point_estimates_df = pd.read_parquet(point_estimate_file)
    actual_columns = [f"{col}_actual" for col in DEPENDENT_COLUMN_NAMES]
    prediction_columns = [f"{col}_prediction" for col in DEPENDENT_COLUMN_NAMES]
    point_estimates_df[actual_columns] = point_estimates_df['actual_value'].apply(
        lambda x: nelson_siegel(np.array(MATURITIES), *x[0]) * 100).apply(pd.Series)
    point_estimates_df[prediction_columns] = point_estimates_df['prediction'].apply(
        lambda x: nelson_siegel(np.array(MATURITIES), *x[0]) * 100).apply(pd.Series)
    # persist the results
    point_estimates_df.to_parquet(f"results/{time_prediction}/{time_prediction}_yield_results.parquet")


def calc_sample_mean_yield(sample_file: str) -> pd.DataFrame:
    """
    Calculate the mean yield from a single sample file.
    Args:
        sample_file: str
            Path to the sample file containing yield indexes.

    """
    date_str = sample_file.split('/')[-1].split('.')[0]
    date_index = pd.Timestamp(date_str)
    df_tmp = pd.read_parquet(sample_file)
    mean_yields = [df_tmp[DEPENDENT_COLUMN_NAMES].mean().to_list()]
    return pd.DataFrame(mean_yields, columns=DEPENDENT_COLUMN_NAMES, index=[date_index])

def calc_mean_yields(time_prediction: str, max_workers: int = 4) -> None:
    """
    Calculate mean yields from the transformed yield samples.
    Args:
        time_prediction: str
            The time prediction identifier (e.g., 'one-day-ahead', 'seven-day-ahead', etc.)
    Returns: None
    """
    sample_files = get_list_of_sample_files(time_prediction=time_prediction, yields=True)
    # concurrently calculate the mean yields
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(calc_sample_mean_yield, sample_files), total=len(sample_files)))
    # concatenate the results
    mean_yields_df = pd.concat(results)
    # persist the results
    mean_yields_df.to_parquet(f"results/{time_prediction}/{time_prediction}_mean_yields.parquet")

def run_pipeline_transformations(time_prediction: str, max_workers: int=4) -> None:
    """
    Run the complete transformation pipeline for a given time prediction.
    Args:
        time_prediction: str
            The time prediction identifier (e.g., 'one-day-ahead', 'seven-day-ahead', etc.)
        max_workers: int
            Maximum number of workers for concurrent processing.
    Returns: None
    """
    # run the pipeline transformations
    # get the list of sample files
    sample_files = get_list_of_sample_files(time_prediction=time_prediction)
    print(f"Transforming {len(sample_files)} sample files for time prediction: {time_prediction}")
    transform_all_samples_to_yields(sample_files=sample_files, time_prediction=time_prediction, max_workers=max_workers)
    print("Transforming point estimates to yields...")
    try:
        transform_point_estimates_to_yields(time_prediction=time_prediction)
    except Exception as e:
        print(f"Error transforming point estimates to yields: {str(e)}")
    print("Calculating mean yields from samples...")
    calc_mean_yields(time_prediction=time_prediction, max_workers=max_workers)
    print(f"Pipeline transformations completed for time prediction: {time_prediction}")

if __name__ == "__main__":
    # pass the time prediction as an argument
    time_prediction = sys.argv[1]
    run_pipeline_transformations(time_prediction=time_prediction, max_workers=8)
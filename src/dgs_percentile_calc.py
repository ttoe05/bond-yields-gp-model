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

DEPENDENT_COLUMN_NAMES = [
    'DGS1MO_future_val', 'DGS3MO_future_val', 'DGS6MO_future_val',
    'DGS1_future_val', 'DGS2_future_val', 'DGS3_future_val', 'DGS5_future_val',
    'DGS7_future_val', 'DGS10_future_val', 'DGS20_future_val', 'DGS30_future_val'
]
def get_list_sample_files(time_prediction: str, percentile: bool = False) -> list[str]:
    """
    Get a list of all sample files for a given time prediction.
    Args:
        time_prediction: str
            The time prediction identifier (e.g., 'one-day-ahead', 'seven-day-ahead', etc.)
    Returns:
        list[str]: List of file paths to sample files.
    """
    # check the time_prediction passed in is valid
    if time_prediction not in ['one-day-ahead', 'seven-day-ahead', 'sixty-day-ahead', 'thirty-day-ahead']:
        raise ValueError(f"Invalid time_prediction: {time_prediction}. Must be one of 'one-day-ahead', 'seven-day-ahead', 'sixty-day-ahead', 'thirty-day-ahead'.")

    if percentile:
        sample_dir = f"results/{time_prediction}/samples/dgs_yields/percentile/"
    else:
        sample_dir = f"results/{time_prediction}/samples/dgs_yields/"
    sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.parquet')]
    return sample_files

def get_percentiles(sample_file: str, time_prediction: str) -> None:
    """
    Calculate percentiles for the dgs yields
    Args:
        sample_file:
        time_prediction:

    Returns:

    """
    sample_df = pd.read_parquet(sample_file)
    for col in DEPENDENT_COLUMN_NAMES:
        sample_df[f"{col}_percentile"] = sample_df[col].rank(pct=True)
    # persist the results
    sample_df.to_parquet(f"results/{time_prediction}/samples/dgs_yields/percentile/{sample_file.split('/')[-1]}")


def calc_all_percentiles(sample_files: list[str], time_prediction: str, max_workers: int=4) -> None:
    """
    Concurrently transform all sample files of NS parameters into yield indexes.
    Args:
        sample_files:
        time_prediction:
        max_workers:
    Returns: None
    """
    # create the yields subdirectory if it doesn't exist
    yields_dir = Path(f'results/{time_prediction}/samples/dgs_yields/percentile/')
    Path(yields_dir).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(get_percentiles, sample_files, repeat(time_prediction)), total=len(sample_files)))


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
    sample_files = get_list_sample_files(time_prediction=time_prediction)
    print(f"Transforming {len(sample_files)} sample files for time prediction: {time_prediction}")
    calc_all_percentiles(sample_files=sample_files, time_prediction=time_prediction, max_workers=max_workers)
    print(f"Pipeline transformations completed for time prediction: {time_prediction}")


if __name__ == "__main__":
    # pass the time prediction as an argument
    try:
        time_prediction = sys.argv[1]
    except IndexError:
        time_prediction = None
    if time_prediction is None:
        # run for all time predictions
        time_prediction_list = [
            'one-day-ahead', 'seven-day-ahead', 'thirty-day-ahead', 'sixty-day-ahead'
            ]
        for time_prediction in time_prediction_list:
            run_pipeline_transformations(time_prediction=time_prediction, max_workers=8)
    else:
        run_pipeline_transformations(time_prediction=time_prediction, max_workers=8)
    print("All transformations completed.")
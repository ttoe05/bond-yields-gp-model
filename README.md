# Nelson-Siegel Bond Yield Forecasting

A Gaussian Process-based forecasting framework for predicting Nelson-Siegel model coefficients using walk-forward validation. This system provides robust time series forecasting of yield curve parameters with uncertainty quantification and automatic kernel selection.

## Features

- **Multiple Kernel Selection**: Automatically evaluates and selects optimal GP kernels (RBF, ExpSineSquared, RationalQuadratic)
- **Uncertainty Quantification**: Provides prediction intervals and full predictive distributions
- **Time Series Validation**: Implements walk-forward validation preserving temporal structure
- **Parallel Training**: Concurrent kernel evaluation for improved performance
- **Configurable Horizons**: Supports multiple prediction time frames (1-day, 7-day ahead, etc.)

## System Architecture

The forecasting system consists of five core modules:

- **`data_loader.py`**: Handles bond yield data loading, validation, and time window generation for walk-forward validation
- **`feature_manager.py`**: Manages feature configurations from YAML files and handles dependent variable selection
- **`gp_models.py`**: Core Gaussian Process ensemble with multiple kernels and parallel training capabilities
- **`walk_forward.py`**: Implements time series walk-forward validation framework with configurable model retraining intervals
- **`main_forecasting.py`**: Orchestrates the complete pipeline from data loading to prediction export

## Installation

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- PyYAML >= 6.0
- tqdm >= 4.64.0

### Python Version

Python 3.8+ recommended for optimal performance with concurrent processing.

## Data Requirements

### Input Data Format
- **Data File**: Parquet format with DateTime-indexed DataFrame containing Nelson-Siegel coefficients and features
- **Feature Configuration**: YAML file specifying feature sets for different prediction horizons
- **Required Structure**: 
  - DateTime index for time series ordering
  - Nelson-Siegel coefficient columns (dependent variables)
  - Feature columns as specified in YAML configuration

### Example Data Structure
```
Date                beta0    beta1    beta2    feature1    feature2    ...
2020-01-01         2.5      -1.2     0.8      1.1         0.3
2020-01-02         2.6      -1.1     0.9      1.2         0.4
...
```

## Configuration

### Key Parameters

Modify the following parameters in `main_forecasting.py` (lines 98-104):

```python
time_prediction = 'seven-day-ahead'        # Forecast horizon
config_file = 'path/to/features.yaml'      # Feature configuration
data_file = 'path/to/data.parquet'         # Input data file
train_window = 2000                        # Training window size
min_train_window = 1000                    # Minimum training window
retrain_interval = 30                      # Model retraining frequency
selection_metric = 'train_cosine_distance' # Kernel selection metric
```

### Configuration Options

- **`time_prediction`**: Available horizons defined in YAML config (e.g., 'one-day-ahead', 'seven-day-ahead')
- **`train_window`**: Number of observations in training window (default: 2000)
- **`min_train_window`**: Minimum required training observations (default: 1000)
- **`retrain_interval`**: Frequency of model retraining in prediction steps (default: 30)
- **`selection_metric`**: Kernel selection criteria options:
  - `'train_cosine_distance'`: Cosine similarity between actual and predicted
  - `'train_euclidean_rmse'`: Root mean squared error
  - `'train_r2_avg'`: Average R-squared across coefficients
  - `'train_r2_flat'`: R-squared on flattened predictions

## Usage

### Basic Usage

Run the forecasting pipeline:

```bash
python main_forecasting.py
```

### Customization Steps

1. **Update File Paths**: Modify lines 99-100 in `main_forecasting.py`:
   ```python
   config_file = '/path/to/your/features_config.yaml'
   data_file = '/path/to/your/bond_data.parquet'
   ```

2. **Set Prediction Horizon**: Change line 98:
   ```python
   time_prediction = 'one-day-ahead'  # or 'seven-day-ahead', etc.
   ```

3. **Adjust Training Parameters**: Modify lines 101-104:
   ```python
   train_window = 1500        # Adjust training window size
   retrain_interval = 20      # Change retraining frequency
   selection_metric = 'train_r2_avg'  # Use different metric
   ```

## Model Details

### Gaussian Process Ensemble

The system implements an ensemble of Gaussian Process regressors with different kernel configurations:

- **RBF Kernel**: Radial Basis Function for smooth, stationary patterns
- **Exponential Sine Squared**: Captures periodic behavior in yield curves  
- **Rational Quadratic**: Flexible kernel combining multiple length scales

### Kernel Selection Process

1. **Parallel Training**: All kernels are trained concurrently using multiprocessing
2. **Performance Evaluation**: Models evaluated using configurable metrics
3. **Automatic Selection**: Best-performing kernel selected for predictions
4. **Uncertainty Quantification**: Full predictive distributions generated

### Walk-Forward Validation

- **Temporal Integrity**: Preserves time series structure during validation
- **Expanding Windows**: Training window grows over time while maintaining maximum size
- **Periodic Retraining**: Models retrained at specified intervals to adapt to regime changes
- **Out-of-Sample Predictions**: Each prediction uses only historical data

## Output Format

### Results Directory Structure
```
results/
└── {time_prediction}/
    ├── {time_prediction}_results.parquet
    ├── samples/
    │   ├── 2023-01-01.parquet
    │   ├── 2023-01-02.parquet
    │   └── ...
    └── forecasting.log
```

### Main Results File
The primary results file contains:
- **date**: Prediction date
- **actual_value**: Actual Nelson-Siegel coefficients
- **prediction**: Point predictions for each coefficient
- **prediction_std**: Standard deviation of predictions
- **best_kernel**: Selected kernel name
- **retrain**: Boolean indicating if model was retrained

### Sample Distributions (Optional)
Individual parquet files containing 1000 samples from the predictive distribution for each prediction date, enabling:
- Full uncertainty quantification
- Risk assessment
- Monte Carlo analysis

### Logging
Comprehensive logging to `forecasting.log` includes:
- Pipeline execution progress
- Model training details
- Kernel selection results
- Performance metrics
- Error handling

## System Flow

1. **Data Loading** (`data_loader.py`): Load and validate parquet data, generate time windows for walk-forward validation

2. **Feature Management** (`feature_manager.py`): Load feature configurations from YAML file, extract relevant features for specified time horizon

3. **Model Training** (`gp_models.py`): Train Gaussian Process ensemble with multiple kernels in parallel, evaluate and select best kernel

4. **Walk-Forward Validation** (`walk_forward.py`): Execute time series validation with periodic model retraining, generate predictions with uncertainty

5. **Results Export** (`main_forecasting.py`): Coordinate all components and export predictions, samples, and summaries

## Performance Considerations

- **Parallel Processing**: Kernel training utilizes multiprocessing for improved speed
- **Memory Management**: Large datasets handled efficiently with chunked processing
- **Computational Complexity**: Training time scales with window size and number of features
- **Retraining Strategy**: Balance between model freshness and computational cost

## Example Workflow

```python
# 1. Configure parameters in main_forecasting.py
time_prediction = 'one-day-ahead'
train_window = 1500

# 2. Run the pipeline
python main_forecasting.py

# 3. Results available in results/one-day-ahead/
# - Point predictions and uncertainty in results file
# - Full predictive samples in samples/ subdirectory
# - Execution details in forecasting.log
```

## Troubleshooting

### Common Issues

1. **Data Loading Errors**: Verify parquet file exists and contains required columns
2. **Feature Configuration**: Ensure YAML file contains specified time_prediction key
3. **Memory Issues**: Reduce train_window size or adjust n_parallel_jobs
4. **Convergence Problems**: Try different selection_metric or kernel configurations

### Log Analysis

Check `forecasting.log` for detailed execution information:
- Data loading statistics
- Feature extraction details  
- Model training progress
- Kernel selection results
- Prediction accuracy metrics
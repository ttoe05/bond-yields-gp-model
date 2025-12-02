# Kernel Ridge Regression (KRR) Experiment Pipeline Plan

## Overview
This plan outlines the implementation of a comprehensive Kernel Ridge Regression experiment pipeline for bond yield forecasting, following the specifications in `claude_md/kRR.md`.

## Components to Implement

### 1. KernelRidgeEnsemble Class (`kernel_ridge_models.py`)

#### Initialization Parameters
- `training_metric`: String - metrics for grid search CV ('mse', 'mse_flat', 'rmse', 'r2_avg', 'r2_flat')
- `random_state`: Integer - for reproducibility
- `n_jobs`: Integer - parallel jobs for grid search

#### Attributes
- `best_model`: None initially, stores best estimator after training
- `kernels`: List of available kernels from analysis
- `residuals`: Training residuals for sampling predictions
- `feature_importance`: Placeholder for feature importance proxy

#### Methods

**1. `create_kernel_configuration()`**
- Returns list of 5 kernels from `analysis/kernel_regression.ipynb`:
  - RBF Kernel: `ConstantKernel * RBF + WhiteKernel`
  - Rational Quadratic: `ConstantKernel * RationalQuadratic + WhiteKernel`
  - ExpSineSquared: `ConstantKernel * ExpSineSquared + WhiteKernel`
  - Additional kernel combinations from analysis
- Sets `self.kernels` attribute

**2. `train_krr(x_train, y_train, alpha_params=None)`**
- Uses `MultiOutputRegressor(KernelRidge())` with `TimeSeriesSplit`
- Grid search over kernels and alpha parameters
- Default alphas: `[1e-2, 1e0, 1e3, 1e4]` (from notebook analysis)
- Sets `self.best_model` to best estimator
- Calculates and stores training residuals in `self.residuals`
- Returns training metrics dictionary with:
  - Performance metrics for selected training_metric
  - Best kernel parameter
  - Best alpha parameter

**3. Metric Functions****
- `calculate_mse()`: Mean squared error
- `calculate_mse_flat()`: Flattened MSE across all outputs
- `calculate_rmse()`: Root mean squared error  
- `calculate_r2_avg()`: Average R² across outputs
- `calculate_r2_flat()`: R² on flattened predictions vs actuals

**4. `predict_val(x)`**
- Point prediction wrapper using `self.best_model.predict(x)`
- Returns predictions for given input

**5. `predict_val_distribution(x, y_train, n_samples=1000)`**
- Point prediction from `predict_val()`
- Calculate mean absolute average of residuals
- Compute covariance matrix of residuals
- Use multivariate normal distribution:
  - Mean: point prediction
  - Covariance: residual covariance matrix
- Return 1000 samples as pandas DataFrame with dependent variables as columns

### 2. WalkForwardValidator Refactors (`archive/walk_forward.py`)

#### Standard Scaling Integration
**Current Issue**: No data scaling before model training/prediction

**Required Changes**:
- Add `StandardScaler` instances for features and targets
- Initialize scalers in `__init__()`:
  ```python
  self.scaler_x = StandardScaler()
  self.scaler_y = StandardScaler()
  ```

**Modifications to `run_single_prediction()`**:
1. **Training Data Scaling**:
   - Scale `x_train` and `y_train` before model training
   - Fit scalers only on first training (initial_run or retrain)
   - Transform subsequent windows using fitted scalers

2. **Prediction Data Scaling**:
   - Transform `x_predict` using fitted `scaler_x`
   - Transform predictions and samples using inverse transform

3. **Inverse Transformation**:
   - Apply `scaler_y.inverse_transform()` to:
     - Point predictions before storage
     - Prediction samples before persistence
     - Actual values for consistent comparison

#### Updated Data Flow:
```python
# Training phase
x_train_scaled = self.scaler_x.fit_transform(x_train)  # First run
y_train_scaled = self.scaler_y.fit_transform(y_train)  # First run
model.train_historical(x=x_train_scaled, y=y_train_scaled)

# Prediction phase  
x_predict_scaled = self.scaler_x.transform(x_predict)
prediction_scaled = model.predict_val(x=x_predict_scaled)
prediction_val = self.scaler_y.inverse_transform(prediction_scaled)

# Samples
samples_scaled = model.predict_val_distribution(x=x_predict_scaled, y=y_train_scaled)
samples = self.scaler_y.inverse_transform(samples_scaled.reshape(-1, samples_scaled.shape[-1]))
```

### 3. BondDataLoader Enhancements (`data_loader.py`)

#### Growing Window Support
**Current Implementation**: Fixed window size with sliding window

**Required Addition**: `get_growing_windows()` method
```python
def get_growing_windows(self, min_window_size: int = 100, step_size: int = 1) -> List[Tuple[int, int]]:
    """
    Generate growing windows for walk-forward validation.
    
    Args:
        min_window_size: Minimum initial window size
        step_size: Step size for expanding windows
        
    Returns:
        List of (start_idx, end_idx) tuples with growing training windows
    """
    windows = []
    n_samples = len(self.data)
    
    # Start with minimum window, grow incrementally
    for end_idx in range(min_window_size, n_samples, step_size):
        start_idx = 0  # Always start from beginning for growing window
        windows.append((start_idx, end_idx))
    
    return windows
```

**Integration with WalkForwardValidator**:
- Add `window_type` parameter: 'sliding' (current) or 'growing'
- Modify `run_walk_forward_validation()` to use appropriate window method

## Implementation Steps

### Step 1: Create KernelRidgeEnsemble Class
1. **File**: Create `kernel_ridge_models.py` 
2. **Dependencies**: sklearn, pandas, numpy, scipy
3. **Key Components**:
   - Initialize with parameters from kRR.md specification
   - Implement 5 kernel configurations from notebook analysis
   - Add MultiOutputRegressor with TimeSeriesSplit for training
   - Implement metric calculation functions
   - Add sampling distribution functionality

### Step 2: Refactor WalkForwardValidator
1. **File**: Modify `archive/walk_forward.py`
2. **Add Scaling Support**:
   - Import StandardScaler
   - Initialize scalers in __init__
   - Modify run_single_prediction for scaling workflow
   - Add inverse transformation for results

### Step 3: Enhance BondDataLoader  
1. **File**: Modify `data_loader.py`
2. **Add Growing Window Method**:
   - Implement get_growing_windows()
   - Add window_type parameter support

### Step 4: Integration Testing
1. **Create Test Script**: Test KRR pipeline end-to-end
2. **Validation**: Compare results with notebook analysis
3. **Performance**: Benchmark against existing GP/Bayesian models

## Expected Outputs

### Model Results
- Point predictions for bond yields
- Prediction samples (1000 per prediction) for uncertainty quantification
- Training metrics comparing kernel performance
- Feature importance proxies

### Validation Metrics
- Comparison across different kernels
- Performance metrics: MSE, RMSE, R², etc.
- Residual analysis for model diagnostics
- Sample distribution analysis

### File Outputs
- Results parquet files: `{time_prediction}_dgs_results.parquet`
- Model summaries: `{time_prediction}_model_summary.parquet` 
- Prediction samples: Individual parquet files per prediction date

## Technical Notes

### Kernel Configurations (from analysis/kernel_regression.ipynb)
1. **RBF**: `ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.5, length_scale_bounds=(1e-7, 1e7)) + WhiteKernel(noise_level=1e-5)`
2. **Rational Quadratic**: `ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=0.1, length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-5, 1e5)) + WhiteKernel(noise_level=1e-5)`
3. **ExpSineSquared**: `ConstantKernel(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=2, periodicity=20.0, length_scale_bounds=(0.01, 10), periodicity_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)`

### Alpha Parameters
Default range from notebook: `[1e-2, 1e0, 1e3, 1e4]`

### Cross-Validation
- TimeSeriesSplit with 3 splits
- Max train size: 1000, test size: 100 (from notebook analysis)

### Dependencies
- scikit-learn: KernelRidge, MultiOutputRegressor, TimeSeriesSplit, GridSearchCV
- numpy: For numerical operations and random sampling
- pandas: Data manipulation and output formatting  
- scipy.stats: For multivariate normal distribution sampling

This plan provides a comprehensive roadmap for implementing the KRR experiment pipeline while integrating seamlessly with the existing walk-forward validation framework.
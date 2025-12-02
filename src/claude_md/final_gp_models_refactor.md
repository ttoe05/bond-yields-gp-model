# GP Modeling Experiments Class - Final Implementation Plan

## Overview
Create a comprehensive `GPModelingExperiments` class for evaluating different types of kernels on bond yield forecasting datasets. This class will extend and refactor the existing `GaussianProcessEnsemble` functionality with better modularity, standardization, and experiment tracking capabilities.

## Class Structure

### Class: `GPModelingExperiments`

#### Initialization
```python
def __init__(self, 
             selection_metric: str = 'train_cosine_distance',
             random_state: int = 42,
             n_jobs: int = 3,
             normalize_features: bool = True,
             normalize_targets: bool = False)
```

**Inputs:**
- `raw_data`: DataFrame containing the complete dataset for analysis
- `prediction_columns`: List of column names to use for predictions (target variables)
- `feature_columns`: List of column names for feature variables
- `selection_metric`: Metric for kernel selection ('train_cosine_distance', 'train_euclidean_rmse', 'train_r2_avg', 'train_r2_flat')
- `random_state`: For reproducibility
- `n_jobs`: Number of parallel jobs
- `normalize_features`: Whether to standardize features
- `normalize_targets`: Whether to normalize target variables

#### Core Methods

##### 1. Feature Transformation
```python
def transform_features(self, x_features: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]
```
**Inputs:**
- `x_features`: DataFrame subset of features to be transformed

**Function:**
- Use StandardScaler to normalize the features
- Store the fitted scaler as class attribute for reuse
- Return tuple of (fitted_scaler_object, transformed_features)

**Returns:**
- `fitted_scaler`: StandardScaler object fitted on the data
- `x_transformed`: Normalized feature matrix

##### 2. Train GP Model
```python
def train_gp_model(self, 
                   x_features: pd.DataFrame, 
                   y_features: pd.Series, 
                   kernel: Any) -> Dict[str, Any]
```
**Inputs:**
- `x_features`: Feature DataFrame
- `y_features`: Target pandas Series
- `kernel`: Defined kernel object

**Function:**
- Apply feature transformation using `transform_features`
- Create MultiOutputRegressor with GaussianProcessRegressor
- Fit the model on transformed features and targets
- Generate predictions on training samples
- Calculate residuals
- Set class attribute `model` to the trained GP model

**Returns:**
- Dictionary containing:
  - `kernel_name`: String identifier
  - `log_marginal_likelihood_value`: Float value
  - `training_residuals`: Residuals from training predictions
  - `model`: Fitted model object
  - `training_metrics`: Performance metrics dictionary

##### 3. Predict
```python
def predict(self, x_features: pd.DataFrame) -> Dict[str, np.ndarray]
```
**Inputs:**
- `x_features`: Feature variables for prediction

**Function:**
- Transform input features using stored scaler
- Call fitted model's predict method with return_std=True and return_cov=True
- Return comprehensive prediction results

**Returns:**
- Dictionary containing:
  - `predictions`: Mean predictions
  - `std_deviations`: Standard deviations
  - `covariance`: Covariance matrix

##### 4. Predict Samples
```python
def predict_samples(self, x_features: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame
```
**Inputs:**
- `x_features`: Feature variables for prediction
- `n_samples`: Number of samples to generate (default: 1000)

**Function:**
- Transform features using stored scaler
- Use residual sampling approach from existing implementation
- Generate bootstrap samples from training residuals
- Convert nd array to pandas DataFrame with proper column names

**Returns:**
- DataFrame with shape (n_samples, n_targets) containing prediction samples

##### 5. Fit-Transform-Predict Pipeline
```python
def fit_transform_predict_pipeline(self,
                                   x_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   kernel: Any,
                                   x_predict: pd.DataFrame,
                                   samples: bool = True) -> tuple
```
**Inputs:**
- `x_train`: Training feature DataFrame
- `y_train`: Training target Series
- `kernel`: Kernel object for GP model
- `x_predict`: Prediction feature DataFrame
- `samples`: Whether to generate prediction samples (default: True)

**Function:**
- Execute complete pipeline in sequence:
  1. Transform features → save to `normalizer`
  2. Train GP model → save metrics to `training_metrics`
  3. Transform prediction features → save to `x_predict_normalized`
  4. Generate predictions → save to `predict_metrics`
  5. Conditionally generate samples → save to `y_samples`

**Returns:**
- Tuple: `(training_metrics, predict_metrics, y_samples)`

#### Enhanced Methods (Extensions)

##### 6. Kernel Configuration Management
```python
def setup_kernel_configurations(self, custom_kernels: Dict[str, Any] = None) -> None
```
- Extend existing `_create_kernel_configurations()` method
- Allow custom kernel definitions
- Include comprehensive kernel library (RBF, Matern, RationalQuadratic, ExpSineSquared, etc.)

##### 7. Experiment Runner
```python
def run_kernel_experiments(self, 
                          x_train: pd.DataFrame, 
                          y_train: pd.Series,
                          validation_data: tuple = None) -> Dict[str, Any]
```
- Execute experiments across all configured kernels
- Support optional validation set evaluation
- Return comprehensive results dictionary with all kernel performance metrics

##### 8. Model Selection and Comparison
```python
def select_best_model(self, metric: str = None) -> str
def get_kernel_rankings(self) -> pd.DataFrame
def plot_kernel_performance(self) -> None
```
- Enhanced model selection with multiple metrics
- Ranking table for all evaluated kernels
- Visualization capabilities for experiment results

##### 9. Experiment Persistence
```python
def save_experiment_results(self, filepath: str) -> None
def load_experiment_results(self, filepath: str) -> None
```
- Save/load complete experiment state including models and metrics
- Support for experiment reproducibility and comparison

## Implementation Architecture

### Class Attributes
- `raw_data`: Input dataset
- `prediction_columns`: Target variable names
- `feature_columns`: Feature variable names
- `normalizer`: Fitted StandardScaler for features
- `target_normalizer`: Optional StandardScaler for targets
- `kernels`: Dictionary of kernel configurations
- `fitted_models`: Dictionary of trained models by kernel name
- `experiment_results`: Comprehensive results storage
- `best_kernel_name`: Selected best performing kernel
- `best_model`: Best performing model object

### Integration with Existing Code
- Inherit metrics computation methods from `GaussianProcessEnsemble`
- Reuse kernel creation logic with enhancements
- Maintain compatibility with existing data loading pipeline
- Preserve multioutput regression capabilities

### Error Handling and Validation
- Input validation for all public methods
- Graceful handling of fitting failures
- Comprehensive logging for experiment tracking
- Proper exception handling with informative messages

## Usage Example
```python
# Initialize experiments
gp_experiments = GPModelingExperiments(
    selection_metric='train_r2_avg',
    random_state=42
)

# Setup kernels (optional custom kernels)
gp_experiments.setup_kernel_configurations()

# Run full experiment pipeline
training_metrics, predict_metrics, samples = gp_experiments.fit_transform_predict_pipeline(
    x_train=X_train,
    y_train=y_train,
    kernel=gp_experiments.kernels['RationalQuadratic'],
    x_predict=X_test,
    samples=True
)

# Or run comprehensive kernel comparison
results = gp_experiments.run_kernel_experiments(X_train, y_train)
best_kernel = gp_experiments.select_best_model()
rankings = gp_experiments.get_kernel_rankings()
```

## Benefits of This Design
1. **Modularity**: Clear separation of concerns with focused methods
2. **Flexibility**: Easy to add new kernels and evaluation metrics
3. **Reproducibility**: Comprehensive experiment tracking and persistence
4. **Scalability**: Support for parallel kernel evaluation
5. **Usability**: Simple pipeline interface with detailed control options
6. **Integration**: Seamless compatibility with existing forecasting workflow
# Bond Yield Forecasting Pipeline Refactoring Plan

## Overview
This document outlines the comprehensive refactoring plan to enhance the bond yield forecasting pipeline with unified configuration, improved model capabilities, and advanced hyperparameter optimization.

## Current State Analysis

### Existing Issues
1. **Fragmented Model Integration**: KernelRidge not integrated into main pipeline despite being most advanced
2. **Inconsistent Training Methods**: GP lacks GridSearch, BayesianRidge lacks cross-validation
3. **Limited Configuration**: Hardcoded parameters, no unified configuration system
4. **Underutilized GP Models**: Most kernels disabled, only RationalQuadratic active
5. **No Scaling Options**: Basic StandardScaler only, no robust alternatives
6. **Inconsistent Interfaces**: Different APIs across model classes
7. **Inconsistent Sampling Methods**: GP models don't use sklearn's built-in posterior sampling
8. **Suboptimal Uncertainty Quantification**: Bayesian Ridge lacks residual-based sampling methodology

## Refactoring Objectives

### Key Sampling Methodology Improvements

#### GP Models: Transition to sklearn's Built-in Posterior Sampling
**Current Problem**: GP models use custom approximations rather than sklearn's theoretically sound `sample_y()` method
**Solution**: Replace custom sampling with `GaussianProcessRegressor.sample_y()` for proper uncertainty quantification
**Benefits**:
- Theoretically correct GP posterior sampling
- Better calibrated uncertainty estimates  
- Consistency with sklearn's intended usage patterns
- Improved model reliability for financial forecasting

#### Bayesian Ridge: Adopt Residual-based Sampling
**Current Problem**: Bayesian Ridge lacks sophisticated uncertainty quantification
**Solution**: Implement KRR-style residual-based sampling using absolute mean of training residuals
**Benefits**:
- Consistent sampling methodology across KRR and Bayesian Ridge
- Empirically-grounded uncertainty estimates
- Computationally efficient compared to Monte Carlo methods
- Proven methodology from successful KRR implementation

### Phase 1: Model Standardization and Enhancement

#### 1.1 Standardize Model Interfaces
**Priority: High**

**Tasks:**
- [ ] Create `BaseEnsembleModel` abstract base class with standardized methods:
  ```python
  class BaseEnsembleModel(ABC):
      @abstractmethod
      def train_historical(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]
      @abstractmethod
      def predict_val(self, x: pd.DataFrame) -> np.ndarray
      @abstractmethod
      def predict_val_distribution(self, x: pd.DataFrame, y: pd.DataFrame, n_samples: int) -> pd.DataFrame
      @abstractmethod
      def get_model_summary(self) -> Dict[str, Any]
      @abstractmethod
      def get_feature_importance_proxy(self, X: pd.DataFrame) -> pd.Series
  ```

- [ ] Update all model classes to inherit from `BaseEnsembleModel`
- [ ] Ensure consistent method signatures across all models
- [ ] Add common validation and error handling

**Files to Modify:**
- `gp_models.py`
- `bayesian_ridge_models.py` 
- `kernel_ridge_models.py`

#### 1.2 Enhance GP Models with Unified Kernel Configurations
**Priority: High**

**Current State:**
- Only RationalQuadratic kernel active
- No GridSearch implementation
- Limited hyperparameter optimization

**Tasks:**
- [ ] Import kernel configurations from `kernel_ridge_models.py`:
  ```python
  def create_kernel_configuration(self) -> List:
      """Use same kernels as KernelRidgeEnsemble for consistency."""
      kernels = [
          # RBF Kernel
          ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.5, 
                                                 length_scale_bounds=(1e-7, 1e7)) + 
          WhiteKernel(noise_level=1e-5),
          
          # Rational Quadratic Kernel  
          ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, 
                                                               alpha=0.1,
                                                               length_scale_bounds=(1e-5, 1e5),
                                                               alpha_bounds=(1e-5, 1e5)) + 
          WhiteKernel(noise_level=1e-5),
          
          # ExpSineSquared Kernel
          # ... (all 5 kernels from KRR)
      ]
  ```

- [ ] Implement GridSearch with TimeSeriesSplit:
  ```python
  def train_gp_with_gridsearch(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
      """Implement GridSearch similar to KernelRidgeEnsemble."""
      param_grid = {'kernel': self.kernels}
      cv = TimeSeriesSplit(n_splits=3, max_train_size=1000, test_size=50)
      # ... rest of GridSearch implementation
  ```

- [ ] Add ensemble option vs single kernel option:
  ```python
  def __init__(self, ensemble_method: str = 'best_single', # or 'ensemble_average'
               kernel_selection: Optional[int] = None):  # specify single kernel index
  ```

- [ ] **Refactor GP Model Sampling to Use sklearn's y_samples**:
  ```python
  def predict_val_distribution(self, x: pd.DataFrame, y: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
      """
      Use sklearn's GaussianProcessRegressor.sample_y() method for proper uncertainty quantification.
      Replace current custom sampling with GP's built-in posterior sampling.
      """
      if self.best_model is None:
          raise ValueError("Model must be trained first.")
      
      samples_list = []
      for estimator in self.best_model.estimators_:
          # Use sklearn's sample_y method for proper GP posterior sampling
          y_samples = estimator.sample_y(x.values, n_samples=n_samples, random_state=self.random_state)
          samples_list.append(y_samples)
      
      # Aggregate samples across outputs
      samples_array = np.column_stack(samples_list)
      samples_df = pd.DataFrame(samples_array, columns=y.columns)
      
      # Apply prediction boundaries if available
      if hasattr(self, 'training_columns') and self.training_columns is not None:
          bounded_samples = self._apply_prediction_boundaries(samples_df.values, list(samples_df.columns))
          samples_df = pd.DataFrame(bounded_samples, columns=samples_df.columns)
      
      return samples_df
  ```

**Benefits of GP y_samples approach:**
- **Theoretical soundness**: Uses proper GP posterior sampling rather than approximations
- **Uncertainty quantification**: Captures model uncertainty correctly through GP framework
- **Consistency**: Matches sklearn's intended GP usage patterns
- **Better calibration**: GP uncertainty estimates are well-calibrated compared to residual-based approximations

**Files to Modify:**
- `gp_models.py`

#### 1.3 Add GridSearch to Bayesian Ridge Models
**Priority: Medium**

**Current State:**
- Manual alpha testing without cross-validation
- Limited hyperparameter search

**Tasks:**
- [ ] Implement TimeSeriesSplit GridSearch similar to KRR:
  ```python
  def train_bayesian_ridge(self, x_train: pd.DataFrame, y_train: pd.DataFrame, 
                          alpha_1_params: List[float] = None,
                          alpha_2_params: List[float] = None,
                          lambda_1_params: List[float] = None,
                          lambda_2_params: List[float] = None):
      """Train with GridSearch cross-validation."""
      if alpha_1_params is None:
          alpha_1_params = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
      # ... create param_grid and run GridSearch
  ```

- [ ] Add proper cross-validation scoring
- [ ] Implement model selection based on CV scores rather than training metrics

- [ ] **Implement Kernel Ridge Sampling Methodology for Bayesian Ridge**:
  ```python
  def predict_val_distribution(self, x: pd.DataFrame, y: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
      """
      Adopt the residual-based sampling approach from KernelRidgeEnsemble.
      Use absolute mean of training residuals as standard deviation for sampling.
      """
      if self.best_model is None:
          raise ValueError("Model must be trained first.")
      
      if self.residuals is None:
          raise ValueError("Residuals not available. Ensure train_historical() was called.")
      
      # Get point prediction
      point_pred = self.predict_val(x)
      mean_prediction_df = pd.DataFrame(data=point_pred, columns=y.columns)
      residuals_df = pd.DataFrame(data=self.residuals, columns=y.columns)
      
      # Sample using residual-based methodology (same as KRR)
      samples_dict = {}
      for pred_col in mean_prediction_df.columns:
          mean_val = mean_prediction_df[pred_col].values[0]
          # Use absolute mean of residuals as standard deviation
          std_val = residuals_df[pred_col].abs().mean()
          samples_dict[pred_col] = np.random.normal(
              loc=mean_val,
              scale=std_val,
              size=n_samples
          )
      
      samples_df = pd.DataFrame(samples_dict)
      
      # Apply boundaries if available
      if hasattr(self, 'training_columns') and self.training_columns is not None:
          bounded_samples = self._apply_prediction_boundaries(samples_df.values, list(samples_df.columns))
          samples_df = pd.DataFrame(bounded_samples, columns=samples_df.columns)
      
      return samples_df
  ```

- [ ] **Add residual tracking to training process**:
  ```python
  def train_historical(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
      # ... existing training code ...
      
      # Calculate and store residuals after training (same as KRR)
      train_predictions = self.best_model.predict(x.values)
      self.residuals = y.values - train_predictions
      
      # Store column names for boundary enforcement
      self.training_columns = list(y.columns)
      
      # Calculate training data boundaries (1st and 99th percentiles)
      self.training_percentiles = {}
      for col in self.training_columns:
          p1 = np.percentile(y[col].dropna(), 1)
          p99 = np.percentile(y[col].dropna(), 99)
          self.training_percentiles[col] = (p1, p99)
          
      # ... rest of training metrics calculation ...
  ```

**Benefits of Residual-based Sampling for Bayesian Ridge:**
- **Consistency**: Same sampling methodology across KRR and Bayesian Ridge models
- **Empirical uncertainty**: Uses actual model residuals to estimate prediction uncertainty
- **Computational efficiency**: Faster than Monte Carlo approaches
- **Proven methodology**: Already tested and validated in KRR implementation

**Files to Modify:**
- `bayesian_ridge_models.py`

### Phase 2: Main Forecasting Pipeline Enhancement

#### 2.1 Create Unified Configuration System
**Priority: High**

**Tasks:**
- [ ] Create configuration schemas:
  ```yaml
  # config/model_configs.yaml
  models:
    gaussian_process:
      ensemble_method: "best_single"  # or "ensemble_average"
      kernel_selection: null  # or specific kernel index
      training_metric: "r2_flat"
      n_jobs: -1
      
    kernel_ridge:
      training_metric: "r2_flat"
      alpha_params: [1e-2, 1e0, 1e3, 1e4]
      n_jobs: -1
      
    bayesian_ridge:
      training_metric: "r2_flat" 
      alpha_1_params: [1e-6, 1e-4, 1e-2, 1.0, 10.0]
      alpha_2_params: [1e-6, 1e-4, 1e-2, 1.0, 10.0]
      
  windowing:
    type: "sliding"  # or "growing"
    window_size: 3000
    min_window_size: 2000
    step_size: 1
    model_retrain_interval: 20
    
  scaling:
    enabled: true
    method: "standard"  # or "robust", "quantile", "minmax"
    feature_scaling: "standard"
    target_scaling: "standard"
  ```

- [ ] Create configuration loader class:
  ```python
  class ConfigManager:
      def load_model_config(self, model_type: str) -> Dict[str, Any]
      def load_windowing_config(self) -> Dict[str, Any] 
      def load_scaling_config(self) -> Dict[str, Any]
      def validate_config(self, config: Dict[str, Any]) -> bool
  ```

**Files to Create:**
- `config/model_configs.yaml`
- `config/windowing_configs.yaml`
- `config/scaling_configs.yaml`
- `config_manager.py`

#### 2.2 Enhanced Main Forecasting Pipeline
**Priority: High**

**Tasks:**
- [ ] Refactor `main_forecasting.py` to support all three models:
  ```python
  class UnifiedForecastingPipeline:
      def __init__(self, config_path: str):
          self.config = ConfigManager(config_path)
          
      def create_model(self, model_type: str):
          """Factory method to create any model type."""
          if model_type == 'gaussian_process':
              return GaussianProcessEnsemble(**self.config.load_model_config('gaussian_process'))
          elif model_type == 'kernel_ridge':
              return KernelRidgeEnsemble(**self.config.load_model_config('kernel_ridge'))
          elif model_type == 'bayesian_ridge':
              return BayesianRidgeEnsemble(**self.config.load_model_config('bayesian_ridge'))
              
      def setup_walk_forward(self, model, data_loader, feature_manager):
          """Setup walk forward with configurable options."""
          windowing_config = self.config.load_windowing_config()
          scaling_config = self.config.load_scaling_config()
          
          return WalkForwardValidator(
              model=model,
              data_loader=data_loader, 
              feature_manager=feature_manager,
              window_type=windowing_config['type'],
              window_size=windowing_config['window_size'],
              use_scaling=scaling_config['enabled'],
              **windowing_config
          )
  ```

- [ ] Add command-line interface:
  ```python
  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument('--model', choices=['gaussian_process', 'kernel_ridge', 'bayesian_ridge'])
      parser.add_argument('--config', default='config/default_config.yaml')
      parser.add_argument('--time-prediction', choices=['one-day-ahead', 'seven-day-ahead', 'thirty-day-ahead', 'sixty-day-ahead'])
      parser.add_argument('--window-type', choices=['sliding', 'growing'])
      parser.add_argument('--scaling', action='store_true')
      # ... parse and run
  ```

**Files to Modify:**
- `main_forecasting.py`

#### 2.3 Advanced Windowing Options
**Priority: Medium**

**Tasks:**
- [ ] Enhance `WalkForwardValidator` with additional windowing strategies:
  ```python
  class WalkForwardValidator:
      def __init__(self, window_type: str = 'sliding',  # 'sliding', 'growing', 'expanding_capped'
                   window_cap: Optional[int] = None,  # max size for expanding windows
                   time_based_splits: bool = False,  # use time-based rather than sample-based splits
                   **kwargs):
  ```

- [ ] Implement expanding with cap:
  ```python
  def get_expanding_capped_windows(self, window_cap: int):
      """Growing windows with maximum size cap."""
      # Implementation for capped expanding windows
  ```

- [ ] Add time-based windowing:
  ```python
  def get_time_based_windows(self, time_window: str = '2Y'):
      """Windows based on time periods rather than sample counts."""
      # Implementation for time-based splits
  ```

**Files to Modify:**
- `walk_forward.py`
- `data_loader.py`

#### 2.4 Advanced Scaling Options
**Priority: Medium**

**Tasks:**
- [ ] Create `ScalingManager` class:
  ```python
  class ScalingManager:
      def __init__(self, feature_method: str = 'standard',
                   target_method: str = 'standard'):
          self.scalers = {
              'standard': StandardScaler(),
              'robust': RobustScaler(), 
              'quantile': QuantileTransformer(),
              'minmax': MinMaxScaler()
          }
          
      def setup_scalers(self, feature_method: str, target_method: str):
          return self.scalers[feature_method], self.scalers[target_method]
  ```

- [ ] Integrate into `WalkForwardValidator`:
  ```python
  def __init__(self, scaling_config: Dict[str, str] = None, **kwargs):
      if scaling_config:
          scaling_manager = ScalingManager()
          self.scaler_x, self.scaler_y = scaling_manager.setup_scalers(
              scaling_config.get('feature_scaling', 'standard'),
              scaling_config.get('target_scaling', 'standard')
          )
  ```

**Files to Create:**
- `scaling_manager.py`

**Files to Modify:**
- `walk_forward.py`

#### 2.5 Model-Specific Results Export System
**Priority: High**

**Current Problem**: Results are stored in generic locations without model-specific organization
**Solution**: Implement organized, model-specific result storage with automatic directory creation

**Tasks:**
- [ ] Create `ResultsManager` class for organized output:
  ```python
  class ResultsManager:
      def __init__(self, base_path: str = "results", time_prediction: str = "seven-day-ahead"):
          self.base_path = Path(base_path)
          self.time_prediction = time_prediction
          
      def get_model_results_path(self, model_type: str, result_type: str = "predictions") -> Path:
          """
          Get model-specific results path with automatic directory creation.
          
          Args:
              model_type: "gaussian_process", "kernel_ridge", or "bayesian_ridge"
              result_type: "predictions", "samples", "metrics", "models", "plots"
          
          Returns:
              Path object for model-specific results
          """
          model_path = self.base_path / model_type / self.time_prediction / result_type
          model_path.mkdir(parents=True, exist_ok=True)
          return model_path
          
      def create_directory_structure(self, model_types: List[str]):
          """Create complete directory structure for all models."""
          result_types = ["predictions", "samples", "metrics", "models", "plots", "logs"]
          
          for model_type in model_types:
              for result_type in result_types:
                  dir_path = self.get_model_results_path(model_type, result_type)
                  logger.info(f"Created directory: {dir_path}")
  ```

- [ ] **Enhance WalkForwardValidator with model-specific exports**:
  ```python
  class WalkForwardValidator:
      def __init__(self, results_manager: ResultsManager, model_type: str, **kwargs):
          self.results_manager = results_manager
          self.model_type = model_type
          
          # Create model-specific directories
          self.predictions_dir = results_manager.get_model_results_path(model_type, "predictions")
          self.samples_dir = results_manager.get_model_results_path(model_type, "samples") 
          self.metrics_dir = results_manager.get_model_results_path(model_type, "metrics")
          self.models_dir = results_manager.get_model_results_path(model_type, "models")
          
      def export_results(self, run_id: str = None):
          """Export all results to model-specific directories."""
          if run_id is None:
              run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
              
          # Export predictions
          results_df = pd.DataFrame(self.results_list)
          results_file = self.predictions_dir / f"{self.model_type}_{self.time_prediction}_results_{run_id}.parquet"
          results_df.to_parquet(results_file, index=False)
          
          # Export model summaries
          model_summary_df = pd.DataFrame(self.model_summaries)
          summary_file = self.metrics_dir / f"{self.model_type}_{self.time_prediction}_summary_{run_id}.parquet"
          model_summary_df.to_parquet(summary_file, index=False)
          
          # Export model artifacts
          model_file = self.models_dir / f"{self.model_type}_{self.time_prediction}_model_{run_id}.pkl"
          joblib.dump(self.model, model_file)
          
          logger.info(f"Exported {self.model_type} results to {self.predictions_dir.parent}")
  ```

- [ ] **Update sample persistence with model-specific paths**:
  ```python
  def run_single_prediction(self, ...):
      # ... existing prediction code ...
      
      # Persist samples if needed with model-specific directory
      if self.persist_samples:
          # Use model-specific samples directory
          sample_file = self.samples_dir / f"{predict_date}.parquet"
          samples_df.to_parquet(sample_file)
  ```

- [ ] **Add configuration-driven export settings**:
  ```yaml
  # config/export_configs.yaml
  export:
    base_path: "results"
    create_directories: true
    export_formats:
      predictions: "parquet"  # or "csv", "json"
      samples: "parquet"
      metrics: "parquet"
      models: "pickle"  # or "joblib"
    
    directory_structure:
      # results/{model_type}/{time_prediction}/{result_type}/
      pattern: "{base_path}/{model_type}/{time_prediction}/{result_type}"
      
    file_naming:
      # Include timestamp and run_id in filenames
      include_timestamp: true
      include_run_id: true
      pattern: "{model_type}_{time_prediction}_{result_type}_{timestamp}"
  ```

- [ ] **Integrate with main forecasting pipeline**:
  ```python
  class UnifiedForecastingPipeline:
      def __init__(self, config_path: str):
          self.config = ConfigManager(config_path)
          self.results_manager = ResultsManager(**self.config.load_export_config())
          
      def run_forecast(self, model_type: str, time_prediction: str):
          # Create model-specific directories
          self.results_manager.create_directory_structure([model_type])
          
          # Setup walk forward with results manager
          walk_forward = WalkForwardValidator(
              results_manager=self.results_manager,
              model_type=model_type,
              time_prediction=time_prediction,
              # ... other parameters
          )
          
          # Run validation and export results
          walk_forward.run_walk_forward_validation()
          walk_forward.export_results()
  ```

**Directory Structure Example:**
```
results/
├── gaussian_process/
│   ├── seven-day-ahead/
│   │   ├── predictions/
│   │   │   ├── gp_seven-day-ahead_results_20241115_143022.parquet
│   │   │   └── gp_seven-day-ahead_results_20241115_150315.parquet
│   │   ├── samples/
│   │   │   ├── 2024-01-15.parquet
│   │   │   └── 2024-01-16.parquet
│   │   ├── metrics/
│   │   │   └── gp_seven-day-ahead_summary_20241115_143022.parquet
│   │   ├── models/
│   │   │   └── gp_seven-day-ahead_model_20241115_143022.pkl
│   │   └── plots/
│   │       └── confidence_intervals_train.png
│   └── thirty-day-ahead/
│       └── ...
├── kernel_ridge/
│   └── seven-day-ahead/
│       └── ...
└── bayesian_ridge/
    └── seven-day-ahead/
        └── ...
```

**Benefits:**
- **Organization**: Clear separation of results by model and time horizon
- **Traceability**: Timestamped files for experiment tracking
- **Automation**: Automatic directory creation prevents errors
- **Consistency**: Standardized naming conventions across all models
- **Scalability**: Easy to add new models or time horizons

**Files to Create:**
- `results_manager.py`
- `config/export_configs.yaml`

**Files to Modify:**
- `walk_forward.py`
- `main_forecasting.py`

### Phase 3: Integration and Testing

#### 3.1 Boundary Enhancement for All Models
**Priority: Medium**

**Tasks:**
- [ ] Add prediction boundaries to GP models (similar to KRR):
  ```python
  # Add to GaussianProcessEnsemble
  def _apply_prediction_boundaries(self, predictions, column_names):
      # Same implementation as KRR
  ```

- [ ] Add prediction boundaries to Bayesian Ridge models
- [ ] Ensure consistent boundary enforcement across all models

**Files to Modify:**
- `gp_models.py`
- `bayesian_ridge_models.py`

#### 3.2 Comprehensive Testing
**Priority: High**

**Tasks:**
- [ ] Create integration tests for all model combinations
- [ ] Test configuration loading and validation
- [ ] Test windowing and scaling combinations
- [ ] Performance benchmarking across models

**Files to Create:**
- `tests/test_unified_pipeline.py`
- `tests/test_model_consistency.py`
- `tests/test_configurations.py`

## Implementation Timeline

### Week 1: Model Standardization
- [ ] Implement `BaseEnsembleModel`
- [ ] Update all model classes to inherit from base
- [ ] Add GridSearch to GP models
- [ ] **Refactor GP sampling to use sklearn's sample_y method**

### Week 2: Configuration System and Sampling Unification
- [ ] Create configuration schemas
- [ ] Implement `ConfigManager`
- [ ] Add GridSearch to Bayesian Ridge
- [ ] **Implement KRR-style residual sampling in Bayesian Ridge models**
- [ ] **Add residual tracking and boundary enforcement to Bayesian Ridge**

### Week 3: Pipeline Enhancement
- [ ] Refactor `main_forecasting.py`
- [ ] Implement model factory
- [ ] Add command-line interface

### Week 4: Advanced Features
- [ ] Enhanced windowing strategies
- [ ] Advanced scaling options
- [ ] Prediction boundaries for all models

### Week 5: Testing and Documentation
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Documentation updates

## Success Criteria

1. **Unified Interface**: All models implement consistent APIs
2. **Configuration-Driven**: All parameters configurable via YAML/JSON
3. **Model Parity**: All models have GridSearch and cross-validation
4. **Feature Completeness**: All models support boundaries, scaling, windowing
5. **Performance**: Maintain or improve prediction accuracy
6. **Maintainability**: Clean, documented, testable code

## Rollback Plan

- Maintain current implementations in `archive/` directory
- Use feature flags to enable/disable new functionality
- Comprehensive backup before major changes
- Staged deployment with fallback options

## Dependencies

- PyYAML for configuration management
- scikit-learn updates for additional scalers
- Enhanced logging for debugging
- Configuration validation schemas

This plan ensures a systematic approach to creating a unified, scalable, and maintainable forecasting pipeline while preserving existing functionality and improving model capabilities.
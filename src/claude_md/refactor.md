# Monte Carlo Simulation Refactor Plan

## Current Issues Identified

### 1. **Feature Manager Logic Issues**
- `get_target_variables()` and `get_actual_variables()` methods are broken
- Missing `self.time` attribute but trying to access it in `get_target_variables()`  
- Method `get_actual_variables()` is commented out but still called in main pipeline
- `get_current_variables()` exists but `get_actual_variables()` is called instead

### 2. **Data Flow Inconsistencies**
- Main pipeline calls `get_actual_variables()` but method doesn't exist
- Confusion between "actual_variables", "target_variables", and "current_variables"
- Data loader parameter mismatch: calls with `current_cols` but expects `actuals`

### 3. **Logic Flow Gaps**
- Missing proper prediction index offset to prevent data leakage
- Window iteration doesn't properly account for prediction horizon
- Sample persistence logic may not align with expected file naming

## Refactor Plan

### Phase 1: Fix Feature Manager Methods

#### 1.1 Fix `FeatureManagerSim` class
```python
# In feature_manager_sim.py
def get_target_variables(self) -> List[str]:
    """Get future value columns for comparison against predictions."""
    return self.features_config['future_values']

def get_actual_variables(self) -> List[str]:
    """Get current yield columns for simulation base values."""  
    return self.features_config['current_values']
```

#### 1.2 Remove broken method references
- Remove `get_current_variables()` method (not used in logical flow)
- Ensure consistent naming: `actual_variables` = current yield values

### Phase 2: Align Data Loading Logic

#### 2.1 Fix parameter naming in `data_loader.py`
```python
# Change load_data method signature
def load_data(self, x: List[str], y: List[str], actuals: List[str]) -> None:
```

#### 2.2 Update main pipeline data loading call
```python
# In main_forecasting_sim.py
actual_columns = self.feature_manager_sim.get_actual_variables()
self.data_loader.load_data(x=features, y=y_variables + target_columns, actuals=actual_columns)
```

### Phase 3: Implement Proper Window Logic

#### 3.1 Fix prediction index calculation in `walk_forward_sim.py`
```python
# Prevent data leakage by ensuring prediction point is beyond training window
def run_walk_forward_validation(self) -> None:
    # Get time horizon offset
    time_offset_map = {
        'one-day-ahead': 1,
        'seven-day-ahead': 7, 
        'thirty-day-ahead': 30,
        'sixty-day-ahead': 60
    }
    offset = time_offset_map[self.time_prediction]
    
    # Clip windows to ensure prediction points exist
    windows = windows[:-offset]  # Remove last 'offset' windows
    
    for train_start_idx, train_end_idx in windows:
        # Prediction point is offset days after training end
        predict_idx = train_end_idx + offset
```

#### 3.2 Update window boundaries
- Ensure training windows don't include future data
- Add proper bounds checking for prediction indices

### Phase 4: Standardize Simulation Flow

#### 4.1 Fix `run_single_prediction` method
```python
def run_single_prediction(self, train_start_idx: int, train_end_idx: int, 
                         predict_idx: int, y_variables: List[str], 
                         target_columns: List[str], features: List[str]) -> Dict:
    
    # Get training data (dependent variables for KDE fitting)
    x_train, y_train = self.data_loader.get_window_data(
        start_idx=train_start_idx,
        end_idx=train_end_idx, 
        target_columns=y_variables,  # DGS*_gain_loss columns
        feature_columns=features
    )
    
    # Get actual future values for comparison
    actual_value = self.data_loader.get_actual_value(
        idx=predict_idx, 
        target_columns=target_columns  # DGS*_future_val columns
    )
    
    # For each dependent variable (gain_loss), fit KDE and simulate
    for dep_var in y_variables:
        # Get current yield value (base for adding changes)
        base_yield_name = dep_var.split('_gain_loss_')[0]  # Extract DGS1MO from DGS1MO_gain_loss_1d
        current_yield = self.data_loader.get_current_yield_value(predict_idx, base_yield_name)
        
        # Fit KDE to historical changes and simulate
        # ... Monte Carlo simulation logic
```

### Phase 5: Ensure Proper Sample Persistence

#### 5.1 Fix sample file naming
```python
# Use prediction date as filename
predict_date = self.data_loader.get_date_for_index(predict_idx)
samples_df.to_parquet(self.sample_dir / f"{predict_date}.parquet")
```

#### 5.2 Align sample data structure
- Ensure samples contain all target yield predictions
- Use consistent column naming for analysis downstream

## Implementation Priority

1. **CRITICAL**: Fix `FeatureManagerSim` methods (Phase 1)
2. **CRITICAL**: Fix data loading parameter mismatch (Phase 2.1, 2.2)  
3. **HIGH**: Implement proper window logic to prevent data leakage (Phase 3)
4. **MEDIUM**: Standardize simulation flow (Phase 4)
5. **LOW**: Sample persistence improvements (Phase 5)

## Validation Steps

After implementing fixes:

1. **Unit Tests**: Verify each FeatureManager method returns correct columns
2. **Data Flow Test**: Ensure no data leakage in window construction
3. **Integration Test**: Run single prediction and verify output structure
4. **Sample Verification**: Confirm sample files contain expected data structure

## Expected Outcomes

- **Data Integrity**: No future information leaks into training windows
- **Consistent Naming**: Clear distinction between dependent variables (gain_loss), target variables (future_val), and current variables (DGS*)
- **Proper Simulation**: KDE fitted on historical changes, applied to current yields
- **Reliable Persistence**: Sample files properly named and structured for analysis

## Breaking Changes

- `FeatureManagerSim.get_target_variables()` will return different columns
- `data_loader.load_data()` parameter name change from `current_cols` to `actuals`
- Window iteration logic will change prediction indices

## Migration Notes

- Existing sample files may need regeneration due to logic fixes
- Any downstream analysis code should verify expected column names
- Performance may improve due to more efficient data handling
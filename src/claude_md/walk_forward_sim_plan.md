# Walk Forward Simulation Plan

## Overview
Create `walk_forward_sim.py` that integrates Monte Carlo simulation logic from `monte_sim.py` into the walk-forward validation framework. Instead of using machine learning models to predict yield changes, this approach will:

1. Use training window data to fit distributions of yield changes
2. Sample from these distributions to generate forecasted changes
3. Add the sampled changes to current yields to get forecasted future yields

## Key Changes from `walk_forward.py`

### 1. Class Modifications
- **New Class Name**: `WalkForwardSimulator` (instead of `WalkForwardValidator`)
- **Remove ML Model Dependencies**: Remove `GaussianProcessEnsemble` and `BayesianRidgeEnsemble` imports and parameters
- **Add Monte Carlo Integration**: Import and integrate `MonteCarloSimulator` from `monte_sim.py`

### 2. Constructor Changes
```python
def __init__(self,
             data_loader: BondDataLoader,
             feature_manager_sim: FeatureManagerSim,  # Use new feature manager
             time_prediction: str,
             persist_samples: bool = True,
             window_size: int = 3000,
             min_window_size: int = 2000,
             step_size: int = 1,
             n_simulations: int = 1000):  # New parameter for Monte Carlo
```

**Removed Parameters**:
- `model`: No longer needed as we're using Monte Carlo simulation
- `model_retrain_interval`: Not applicable for distribution-based approach
- `n_parallel_jobs`: Simplified approach doesn't require this

**Added Parameters**:
- `n_simulations`: Number of Monte Carlo simulations to run

### 3. Core Logic Changes

#### 3.1 Training Window Processing
Instead of training ML models, the system will:
- Calculate historical yield changes within the training window
- For each bond yield (DGS1MO, DGS3MO, etc.), extract the corresponding change series
- Use the change data to fit KDE distributions via `MonteCarloSimulator`

#### 3.2 Prediction Logic Replacement
Replace the ML prediction logic in `run_single_prediction()`:

**Current Approach (ML-based)**:
```python
# Train model
self.model.train_historical(x=x_train, y=y_train)
# Get prediction
prediction_val = self.model.predict_val(x=x_predict)
```

**New Approach (Monte Carlo-based)**:
```python
# For each dependent variable, fit distribution and simulate
predictions = {}
prediction_samples = {}

for dep_var in target_columns:
    # Extract historical changes for this yield
    historical_changes = y_train[dep_var].values
    
    # Get current yield value (base for adding changes)
    current_yield = self.data_loader.get_current_yield_value(predict_idx, dep_var)
    
    # Create Monte Carlo simulator
    mc_sim = MonteCarloSimulator(num_simulations=self.n_simulations)
    
    # Fit and simulate
    simulated_yields = mc_sim.fit_and_simulate(historical_changes, current_yield)
    
    # Store mean prediction and all samples
    predictions[dep_var] = np.mean(simulated_yields)
    prediction_samples[dep_var] = simulated_yields.flatten()
```

### 4. Data Access Modifications

#### 4.1 New Data Loader Method
Need to add method to `data_loader.py`:
```python
def get_current_yield_value(self, idx: int, yield_column: str) -> float:
    """Get the current yield value for a specific bond at given index."""
    # Extract base yield name (e.g., 'DGS1MO' from 'DGS1MO_gain_loss_1d')
    base_yield = yield_column.split('_gain_loss_')[0]
    return self.data.iloc[idx][base_yield]
```

#### 4.2 Feature Manager Integration
- Use `feature_manager_sim.py` instead of `feature_manager.py`
- Call `get_dependent_variables(time_prediction)` instead of `get_dependent_variables()`

### 5. Results Structure Changes

#### 5.1 Simplified Result Dictionary
Remove ML-specific fields:
```python
result = {
    'date': predict_date,
    'actual_value': list(actual_value.to_numpy()),
    'prediction': list(prediction_means),
    'simulation_stats': {
        'mean_changes': mean_changes_dict,
        'std_changes': std_changes_dict,
        'num_simulations': self.n_simulations
    }
}
```

#### 5.2 Remove Model Summary Logic
- Remove `model_summary` generation and tracking
- Remove `feature_importance` calculations
- Remove model retraining logic and counters

### 6. Sample Persistence Updates

Modify sample persistence to store Monte Carlo simulation results:
```python
# Create samples DataFrame with all simulations for all yields
samples_data = {}
for dep_var in target_columns:
    samples_data[dep_var] = prediction_samples[dep_var]

samples_df = pd.DataFrame(samples_data)
samples_df.to_parquet(self.sample_dir / f"{predict_date}.parquet")
```

### 7. Method Removals/Simplifications

#### Remove:
- Model retraining logic (`model_retrain_counter`, `initial_run`, etc.)
- Feature importance tracking
- Model summary generation
- Complex ML model management

#### Simplify:
- `run_single_prediction()`: Focus only on Monte Carlo simulation
- Constructor: Remove ML model parameters
- Results export: Remove model summary export

### 8. Directory Structure Updates

Update sample directory structure:
```python
self.sample_dir = Path(f'results/{self.time_prediction}/samples/monte_carlo')
```

### 9. Dependencies and Imports

#### Add:
```python
from monte_sim import MonteCarloSimulator
from feature_manager_sim import FeatureManagerSim
```

#### Remove:
```python
from gp_models import GaussianProcessEnsemble
from bayesian_ridge_models import BayesianRidgeEnsemble
```

## Implementation Steps

1. **Create base structure**: Copy `walk_forward.py` to `walk_forward_sim.py`
2. **Update imports**: Replace ML model imports with Monte Carlo imports
3. **Modify constructor**: Remove ML parameters, add simulation parameters
4. **Rewrite prediction logic**: Replace ML predictions with Monte Carlo simulation
5. **Update data access**: Add current yield value retrieval method
6. **Simplify results**: Remove ML-specific result fields
7. **Test integration**: Ensure proper integration with `feature_manager_sim.py`
8. **Update sample persistence**: Store Monte Carlo simulation results

## Expected Benefits

1. **Simpler Architecture**: No complex ML model management
2. **Direct Yield Forecasting**: Forecasts actual yield values instead of changes
3. **Distribution-Based Uncertainty**: Natural uncertainty quantification through Monte Carlo
4. **Faster Execution**: No model training overhead
5. **Interpretable Results**: Clear connection between historical changes and forecasts

## Key Validation Points

1. Ensure historical yield changes are correctly extracted
2. Verify current yield values are properly retrieved
3. Confirm Monte Carlo samples are correctly generated
4. Validate that forecasted yields are reasonable (current_yield + sampled_change)
5. Test sample persistence and result export functionality
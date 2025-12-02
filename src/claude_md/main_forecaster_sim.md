# Main Forecaster Simulation Plan

## Overview
Create `main_forecaster_sim.py` that serves as the main orchestrator for Monte Carlo simulation-based bond yield forecasting. This file will leverage `walk_forward_sim.py` to generate simulation samples of DGS (Daily Treasury Security) values using distribution-based forecasting instead of machine learning models.

## Key Differences from `main_forecasting.py`

### 1. Class Name and Purpose
- **New Class**: `YieldSimulationPipeline` (instead of `YieldForecastingPipeline`)
- **Purpose**: Orchestrate Monte Carlo simulation-based forecasting instead of ML model-based forecasting
- **Output**: Generate simulation samples of future yield values using historical distribution fitting

### 2. Import Changes

#### Remove ML-Related Imports:
```python
# Remove these imports
from gp_models import GaussianProcessEnsemble
from bayesian_ridge_models import BayesianRidgeEnsemble
from walk_forward import WalkForwardValidator
from feature_manager import FeatureManager
```

#### Add Simulation-Related Imports:
```python
# Add these imports
from walk_forward_sim import WalkForwardSimulator
from feature_manager_sim import FeatureManagerSim
from monte_sim import MonteCarloSimulator
```

### 3. Constructor Modifications

#### Remove ML-Specific Parameters:
```python
# Remove parameters:
# - model_name: No ML models to select
# - selection_metric: Not applicable for distribution-based approach
# - n_jobs: Simplified parallel processing not needed
# - retrain_interval: Not applicable for Monte Carlo simulation
```

#### Add Simulation-Specific Parameters:
```python
def __init__(self, time_prediction: str, config_file: str, data_file: str,
             train_window: int, min_train_window: int, 
             n_simulations: int = 1000,
             persist_samples: bool = True) -> None:
```

**New Parameters**:
- `n_simulations`: Number of Monte Carlo simulations per prediction
- `persist_samples`: Whether to save individual simulation samples

#### Updated Constructor Logic:
```python
self.time_prediction = time_prediction
self.train_window = train_window
self.min_train_window = min_train_window
self.n_simulations = n_simulations
self.persist_samples = persist_samples

# Initialize simulation components
self.feature_manager_sim = FeatureManagerSim(features_config_path=config_file)
self.data_loader = BondDataLoader(data_path=data_file)

logger.info(
    f"Initialized simulation pipeline for time prediction {self.time_prediction} "
    f"with train window {self.train_window}, min train window {self.min_train_window}, "
    f"and {self.n_simulations} simulations per prediction"
)
```

### 4. Pipeline Method Updates

#### Rename Method:
- `run_pipeline()` → `run_simulation_pipeline()`

#### Updated Pipeline Logic:
```python
def run_simulation_pipeline(self) -> None:
    """
    Run the complete Monte Carlo simulation pipeline.
    """
    logger.info("Starting yield simulation pipeline")

    # Get features for the specified time prediction
    features = self.feature_manager_sim.get_features_for_time_pred(time_prediction=self.time_prediction)
    y_variables = self.feature_manager_sim.get_dependent_variables(self.time_prediction)
    target_columns = self.feature_manager_sim.get_target_variables()
    
    self.data_loader.load_data(x=features, y=y_variables + target_columns)
    logger.info(f"Using {len(features)} features for time prediction {self.time_prediction}")
    logger.info(f"Simulating {len(y_variables)} dependent variables")

    # Set up walk-forward simulator
    wf_simulator = WalkForwardSimulator(
        data_loader=self.data_loader,
        feature_manager_sim=self.feature_manager_sim,
        time_prediction=self.time_prediction,
        window_size=self.train_window,
        min_window_size=self.min_train_window,
        n_simulations=self.n_simulations,
        persist_samples=self.persist_samples
    )

    # Execute walk-forward simulation
    wf_simulator.run_walk_forward_validation()
    wf_simulator.export_results(filepath=f'results/{self.time_prediction}/simulation/')

    logger.info("Yield simulation pipeline completed")
```

### 5. Main Execution Updates

#### Updated Configuration:
```python
if __name__ == "__main__":
    start_time = time.time()
    
    # Simulation-specific configuration
    config_file = 'data/features_selected4.yaml'  # Use simulation config
    train_window = 3000  # Larger window for better distribution fitting
    min_train_window = 2000  # Larger minimum window
    n_simulations = 1000  # Number of Monte Carlo simulations
    persist_samples = True  # Save simulation samples for analysis
    
    # File mappings for different time horizons
    file_num = [1, 7, 30, 60]
    time_prediction_list = [
        'one-day-ahead', 'seven-day-ahead', 'thirty-day-ahead', 'sixty-day-ahead'
    ]
    
    for day, time_prediction in zip(file_num, time_prediction_list):
        data_file = f"data/bond_yields_train_shifted_{day}_sim.parquet"  # Use simulation data files
        
        logger.info(f"Running simulation pipeline for {time_prediction} using data file {data_file}")
        
        pipeline = YieldSimulationPipeline(
            time_prediction=time_prediction,
            config_file=config_file,
            data_file=data_file,
            train_window=train_window,
            min_train_window=min_train_window,
            n_simulations=n_simulations,
            persist_samples=persist_samples
        )
        
        pipeline.run_simulation_pipeline()

    end_time = time.time()
    hours = (end_time - start_time) / 3600
    logger.info(f"Total simulation execution time hours: {hours:.2f}")
```

### 6. Key Functional Differences

#### Data Flow:
1. **Load Configuration**: Uses `FeatureManagerSim` to get simulation-specific features
2. **Load Data**: Loads both dependent variables (yield changes) and target variables (actual yields)
3. **Initialize Simulator**: Creates `WalkForwardSimulator` with Monte Carlo parameters
4. **Run Simulation**: Executes walk-forward validation using distribution fitting
5. **Export Results**: Saves simulation results and samples

#### Output Structure:
- **Main Results**: Date, actual values, predicted means, simulation statistics
- **Sample Files**: Individual Monte Carlo simulation samples for uncertainty analysis
- **Directory Structure**: `results/{time_prediction}/simulation/`

### 7. Configuration File Requirements

The simulation pipeline expects `features_selected4.yaml` with structure:
```yaml
future_values: [list of target yield columns]
one-day-ahead:
  dependent_variables: [list of 1-day change columns]
  max_features: [list of feature columns]
seven-day-ahead:
  dependent_variables: [list of 7-day change columns]
  max_features: [list of feature columns]
# ... additional time horizons
```

### 8. Data File Requirements

Simulation-specific data files with naming pattern:
- `bond_yields_train_shifted_1_sim.parquet`
- `bond_yields_train_shifted_7_sim.parquet`
- `bond_yields_train_shifted_30_sim.parquet`
- `bond_yields_train_shifted_60_sim.parquet`

These files should contain:
- **Base yield columns**: DGS1MO, DGS3MO, DGS6MO, etc.
- **Change columns**: DGS1MO_gain_loss_1d, DGS3MO_gain_loss_7d, etc.
- **Feature columns**: All predictor variables

### 9. Error Handling and Validation

#### Add Validation Checks:
```python
def _validate_configuration(self) -> None:
    """Validate simulation configuration."""
    if self.n_simulations < 100:
        logger.warning("Low number of simulations may produce unstable results")
    
    if self.train_window < 1000:
        logger.warning("Small training window may not capture distribution properly")
    
    # Validate data file exists
    if not Path(self.data_file).exists():
        raise FileNotFoundError(f"Simulation data file not found: {self.data_file}")
```

### 10. Logging Enhancements

#### Add Simulation-Specific Logging:
```python
# Configure logging for simulation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_forecasting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
```

#### Enhanced Progress Reporting:
- Log number of simulations per prediction
- Log sample persistence status
- Log distribution fitting statistics
- Report simulation execution time per time horizon

### 11. Expected Benefits

1. **Faster Execution**: No model training overhead
2. **Natural Uncertainty**: Monte Carlo samples provide confidence intervals
3. **Direct Interpretation**: Clear relationship between historical changes and forecasts
4. **Scalable Simulations**: Easy to adjust number of simulations based on computational resources
5. **Comprehensive Sampling**: Detailed simulation samples available for post-analysis

### 12. Integration Points

The `main_forecaster_sim.py` will integrate with:
- `walk_forward_sim.py`: Core simulation engine
- `feature_manager_sim.py`: Feature configuration management
- `monte_sim.py`: Monte Carlo simulation implementation
- `data_loader.py`: Data access (with new `get_current_yield_value` method)

### 13. Usage Example

```bash
python main_forecaster_sim.py
```

This will run Monte Carlo simulations for all time horizons (1, 7, 30, 60 days ahead) and generate:
- Simulation results in `results/{time_prediction}/simulation/`
- Individual sample files in `results/{time_prediction}/samples/monte_carlo/`
- Comprehensive logs in `simulation_forecasting.log`
# Exponential Weighted Mean (EWM) Integration Plan for Monte Carlo Simulation

## Overview
Enhance the current Monte Carlo simulation in `monte_sim.py` to incorporate exponential weighted mean (EWM) so that more recent changes in bond yields are given higher weights when generating samples. This will make the model more responsive to recent market conditions.

## Current Implementation Analysis
The current `MonteCarloSimulator` class in `monte_sim.py`:
- Uses simple mean and variance calculations (lines 33-35)
- Fits a KDE model using all historical data equally weighted (line 36)
- Generates samples based on uniform statistical properties of the entire dataset

## Implementation Plan

### 1. Add EWM Parameters to Constructor
- Add `ewm_alpha` parameter to control the decay rate (default: 0.1)
- Add `use_ewm` boolean flag to enable/disable EWM (default: True)
- Store these as instance variables for use in fitting

### 2. Modify the `fit()` Method
**Current behavior**: 
- Calculates simple mean: `self.mean = y.mean()`
- Calculates simple variance: `self.variance = y.var()`

**Enhanced behavior**:
- Ensure Time series is sorted properly
- Calculate EWM mean: `self.mean = y.ewm(alpha=self.ewm_alpha).mean().iloc[-1]`
- Calculate EWM variance using the formula: `Var_EWM = EWM((y - EWM_mean)²)`
- Optionally weight the KDE fitting process to emphasize recent observations

### 3. Implement EWM Variance Calculation
Create a helper method `_calculate_ewm_variance()`:
- Calculate EWM mean first
- Compute squared deviations from EWM mean
- Apply EWM to the squared deviations
- Return the final EWM variance

### 4. Enhance KDE Fitting with Weights
Two approaches to consider:
1. **Weighted KDE**: Modify the KDE fitting to accept sample weights based on recency
2. **Recent Data Subset**: Use only the most recent N observations for KDE fitting

### 5. Update the `simulate()` Method
- Use EWM-based drift calculation: `drift = self.ewm_mean - (0.5 * self.ewm_variance)`
- Use EWM-based standard deviation: `sqrt(self.ewm_variance)`
- Maintain the existing sampling logic with the new statistics

### 6. Add Configuration Methods
- `set_ewm_alpha(alpha)`: Allow runtime adjustment of the EWM alpha parameter
- `toggle_ewm(use_ewm)`: Enable/disable EWM functionality
- `get_ewm_stats()`: Return current EWM mean, variance, and standard deviation for debugging

### 7. Backward Compatibility
- Ensure the existing interface remains unchanged
- Add optional parameters with sensible defaults
- Provide a fallback to simple statistics if EWM fails

## Technical Implementation Details

### EWM Alpha Selection
- Alpha = 0.1 gives more weight to recent observations (10-day effective window)
- Alpha = 0.05 provides moderate weighting (20-day effective window)
- Alpha = 0.02 provides gentle weighting (50-day effective window)

### Data Requirements
- Ensure the input data `y` is a pandas Series or DataFrame for EWM functionality
- Handle numpy arrays by converting to pandas Series temporarily
- Maintain original data types in outputs

### Error Handling
- Handle cases where insufficient data exists for EWM calculations
- Provide meaningful error messages for invalid alpha values
- Gracefully fallback to simple statistics if EWM computation fails

## Files to Modify
1. **monte_sim.py**: Main implementation file
   - Update `MonteCarloSimulator.__init__()`
   - Modify `MonteCarloSimulator.fit()`
   - Update `MonteCarloSimulator.simulate()`
   - Add helper methods for EWM calculations

## Testing Considerations
- Test with different alpha values to observe sensitivity
- Compare EWM vs simple statistics performance on recent data
- Validate that predictions are more responsive to recent yield changes
- Ensure backward compatibility with existing code

## Expected Benefits
- More responsive predictions during periods of changing market conditions
- Better capture of recent volatility patterns
- Improved model performance during market regime changes
- Maintained statistical rigor with exponential weighting framework
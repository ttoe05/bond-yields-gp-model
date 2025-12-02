#!/usr/bin/env python3
"""
Test script for EWM implementation in Monte Carlo simulator.
"""

import numpy as np
import pandas as pd
from monte_sim import MonteCarloSimulator

def test_ewm_implementation():
    """Test the EWM functionality with sample data."""
    
    # Create sample time series data (simulating yield changes)
    np.random.seed(42)
    n_points = 100
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    
    # Create a trending time series with recent volatility
    base_trend = np.linspace(0.02, 0.025, n_points)
    noise = np.random.normal(0, 0.001, n_points)
    
    # Add more volatility in recent periods
    recent_volatility = np.zeros(n_points)
    recent_volatility[-20:] = np.random.normal(0, 0.002, 20)
    
    yield_data = base_trend + noise + recent_volatility
    yield_series = pd.Series(yield_data, index=dates)
    
    print("Testing EWM Implementation in Monte Carlo Simulator")
    print("=" * 50)
    
    # Test 1: Create simulator with EWM enabled
    print("\n1. Testing EWM-enabled simulator")
    sim_ewm = MonteCarloSimulator(num_simulations=1000, ewm_alpha=0.1, use_ewm=True)
    sim_ewm.fit(yield_series)
    
    print(f"EWM Stats: {sim_ewm.get_ewm_stats()}")
    
    # Test 2: Create simulator with EWM disabled
    print("\n2. Testing EWM-disabled simulator")
    sim_simple = MonteCarloSimulator(num_simulations=1000, use_ewm=False)
    sim_simple.fit(yield_series)
    
    print(f"Simple Stats: {sim_simple.get_ewm_stats()}")
    
    # Test 3: Compare simulations
    current_yield = 0.025
    predictions_ewm = sim_ewm.simulate(current_yield)
    predictions_simple = sim_simple.simulate(current_yield)
    
    print("\n3. Simulation Results Comparison")
    print(f"EWM Predictions - Mean: {np.mean(predictions_ewm):.6f}, Std: {np.std(predictions_ewm):.6f}")
    print(f"Simple Predictions - Mean: {np.mean(predictions_simple):.6f}, Std: {np.std(predictions_simple):.6f}")
    
    # Test 4: Test configuration methods
    print("\n4. Testing configuration methods")
    sim_ewm.set_ewm_alpha(0.2)
    print(f"New alpha: {sim_ewm.ewm_alpha}")
    
    sim_ewm.toggle_ewm(False)
    print(f"EWM disabled: {not sim_ewm.use_ewm}")
    
    sim_ewm.toggle_ewm(True)
    print(f"EWM re-enabled: {sim_ewm.use_ewm}")
    
    # Test 5: Test with numpy array input
    print("\n5. Testing with numpy array input")
    sim_numpy = MonteCarloSimulator(ewm_alpha=0.15)
    sim_numpy.fit(yield_data)  # numpy array instead of pandas series
    
    print(f"Numpy input stats: Mean={sim_numpy.ewm_mean:.6f}, Variance={sim_numpy.ewm_variance:.8f}")
    
    print("\n✅ All tests completed successfully!")
    
    return sim_ewm, sim_simple

if __name__ == "__main__":
    test_ewm_implementation()
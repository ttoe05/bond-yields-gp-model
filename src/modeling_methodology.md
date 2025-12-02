# Risk Modeling Methodology Plan

## Overview
This document outlines the methodology behind `krr_run.py` and `walk_forward.py` working together to conduct experiments in predicting future index yields through a walk-forward validation framework.

## Key Components

### 1. krr_run.py - Experiment Configuration
- **Purpose**: Sets up and executes a single walk-forward validation experiment
- **Key Components**:
  - Feature and target variable configuration using FeatureManager classes
  - Data loading for specific time horizons (7, 30, 60 day ahead forecasts)
  - Model initialization (KernelRidgeEnsemble) with specific hyperparameters
  - Walk-forward validation setup with window and retraining parameters

### 2. walk_forward.py - Validation Framework
- **Purpose**: Implements the walk-forward validation methodology for time series forecasting
- **Core Functionality**:
  - Sliding/growing window validation preserving temporal order
  - Model retraining at specified intervals
  - Scaling and data preprocessing
  - Prediction distribution generation using Monte Carlo sampling
  - Results persistence and model tracking

## Methodology Description for Final Report

### Experimental Design
The risk modeling methodology employs a walk-forward validation framework to evaluate kernel ridge regression models for bond yield forecasting. This approach ensures temporal integrity by training on historical data and testing on future periods, mimicking real-world forecasting scenarios.

### Model Training and Validation Process

**1. Data Preparation**
- Time series data is organized with features at time t predicting yields at t+m (where m ∈ {7, 30, 60} days)
- Features include macroeconomic indicators, yield curve variables, and technical transformations
- StandardScaler applied when enabled to normalize feature and target distributions

**2. Walk-Forward Validation Framework**
- Growing window approach: training window expands over time, starting from minimum window size
- Model retraining occurs at specified intervals (default: 31 trading days) to adapt to market regime changes
- Each prediction step uses only information available at prediction time to prevent look-ahead bias

**3. Kernel Ridge Regression Ensemble**
- Multiple kernel functions tested (RBF, Rational Quadratic, Exponential Sine Squared) to capture different market dynamics
- Alpha regularization parameter tuning to balance bias-variance tradeoff
- Best performing kernel-alpha combination selected based on validation metrics

**4. Risk Distribution Generation**
- Point predictions provide expected yields E[y|X]
- Monte Carlo sampling from model posterior generates prediction distributions
- 1000 samples per prediction provide uncertainty quantification for risk assessment
- Prediction boundaries applied to ensure realistic yield ranges

**5. Results and Model Tracking**
- Prediction accuracy tracked through multiple performance metrics
- Model summaries capture hyperparameter selections and feature importance
- Sample distributions persisted for downstream risk analysis and portfolio optimization

This methodology enables robust evaluation of bond yield forecasting models while generating the prediction distributions necessary for quantitative risk management.
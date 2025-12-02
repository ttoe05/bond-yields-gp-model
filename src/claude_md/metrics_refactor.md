# DGS Percentile Calculator: Smoothing & Time-Dependent Sample Generation Plan

## Overview
This plan outlines the enhancement of `dgs_percentile_calc.py` to add time-series smoothing functionality and generate new samples from smoothed predictions using univariate normal distributions for each yield curve point.

## Current State Analysis

### Existing Infrastructure
- **File Structure**: Sample files organized by time prediction periods (`one-day-ahead`, `seven-day-ahead`, etc.)
- **Data Format**: Each sample file contains 1000 samples x 11 yield columns (DGS1MO to DGS30)
- **Processing**: Currently calculates percentiles for each yield column individually
- **Concurrency**: Uses ThreadPoolExecutor for parallel processing

### Data Characteristics
- **Time Series Nature**: Files are dated (YYYY-MM-DD.parquet) representing prediction dates
- **Yield Columns**: 11 treasury yields from 1-month to 30-year maturities
- **Sample Size**: 1000 samples per prediction date
- **Time Range**: Approximately 2012-2024 (12+ years of data)

## Proposed Enhancement: Smoothed Prediction Samples

### 1. Core Functionality Requirements

#### A. Data Ingestion & Statistics Calculation
- Read all sample files for a given time prediction period chronologically
- Calculate rolling statistics (mean, standard deviation) for each yield across all dates
- Implement time-dependent windowing for local statistics computation
- Store aggregated statistics in memory-efficient data structures

#### B. Time-Series Smoothing Implementation
- **Exponential Moving Average (EMA)**: Apply smoothing to both mean and standard deviation time series
- **Smoothing Parameters**: Configurable half-life/alpha for different forecast horizons
- **Temporal Consistency**: Ensure smoothed values maintain yield curve shape constraints
- **Boundary Handling**: Preserve early time series values where insufficient history exists

#### C. Sample Generation Engine
- Generate new samples using smoothed mean and standard deviation for each yield
- Apply univariate normal distribution: `N(smoothed_mean, smoothed_std)`
- Maintain temporal correlation structure within generated samples
- Enforce yield curve constraints (non-negativity, term structure reasonableness)

### 2. Technical Implementation Plan

#### Phase 1: Enhanced Data Loading System
```python
class SmoothedSampleGenerator:
    def __init__(self, time_prediction: str, smoothing_params: dict):
        self.time_prediction = time_prediction
        self.smoothing_alpha = smoothing_params.get('alpha', 0.1)
        self.min_periods = smoothing_params.get('min_periods', 10)
        
    def load_chronological_samples(self) -> pd.DataFrame:
        """Load all samples sorted by date with multi-index (date, sample_id)"""
        
    def calculate_rolling_statistics(self, samples_df: pd.DataFrame) -> dict:
        """Calculate date-wise mean and std for each yield column"""
```

#### Phase 2: Time-Series Smoothing Engine
```python
    def apply_exponential_smoothing(self, statistics: dict) -> dict:
        """Apply EMA smoothing to mean and std time series"""
        
    def smooth_yield_curve_constraints(self, smoothed_stats: dict) -> dict:
        """Ensure smoothed statistics maintain yield curve properties"""
```

#### Phase 3: Sample Generation & Output
```python
    def generate_smoothed_samples(self, smoothed_stats: dict, n_samples: int = 1000) -> pd.DataFrame:
        """Generate new samples from smoothed distributions"""
        
    def save_smoothed_samples(self, samples_df: pd.DataFrame, output_path: str) -> None:
        """Save generated samples with date-based file naming"""
```

### 3. Directory Structure & Output Format

#### A. Output Directory Structure
```
results/{time_prediction}/samples/smoothed_prediction_samples/
├── 2012-05-26_smoothed.parquet
├── 2012-05-27_smoothed.parquet
├── ...
└── 2024-12-24_smoothed.parquet
```

#### B. File Naming Convention
- **Pattern**: `YYYY-MM-DD_smoothed.parquet`
- **Content**: Same column structure as original samples (11 yield columns)
- **Metadata**: Include smoothing parameters in parquet metadata

### 4. Enhanced Pipeline Integration

#### A. Modified Pipeline Functions
```python
def run_smoothing_pipeline(time_prediction: str, 
                          smoothing_params: dict = None,
                          max_workers: int = 4) -> None:
    """Run complete smoothing and sample generation pipeline"""
    
def get_smoothed_sample_files(time_prediction: str) -> list[str]:
    """Get list of generated smoothed sample files"""
```

#### B. CLI Integration
```python
if __name__ == "__main__":
    # Add smoothing flag and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--min_periods', type=int, default=10)
```

### 5. Time-Dependency Considerations

#### A. Smoothing Window Strategy
- **Adaptive Windows**: Smaller windows for early dates, expanding as more data becomes available
- **Forward-Looking Smoothing**: Option to include future information for backtesting scenarios
- **Cross-Validation**: Validation framework to optimize smoothing parameters

#### B. Yield Curve Constraints
- **Term Structure**: Maintain reasonable yield curve shapes (normal, inverted, flat)
- **No-Arbitrage**: Ensure generated yields don't violate basic financial constraints
- **Volatility Bounds**: Cap maximum volatility to prevent unrealistic scenarios

### 6. Performance & Scalability

#### A. Memory Management
- **Streaming Processing**: Process files in chunks for large datasets
- **Efficient Storage**: Use parquet compression and column-wise storage
- **Garbage Collection**: Explicit memory cleanup for long-running processes

#### B. Parallel Processing Strategy
```python
def parallel_smoothing_worker(date_chunk: list[str], 
                             smoothing_params: dict) -> dict:
    """Worker function for parallel date processing"""
```

### 7. Configuration & Parameters

#### A. Smoothing Configuration File
```yaml
# smoothing_config.yaml
smoothing_parameters:
  alpha: 0.1  # EMA decay parameter
  min_periods: 10  # Minimum observations for smoothing
  yield_floor: 0.0001  # Minimum yield value (0.01%)
  volatility_cap: 0.05  # Maximum daily volatility (5%)
  
output_settings:
  n_samples: 1000  # Samples per date
  compression: 'snappy'  # Parquet compression
  preserve_metadata: true
```

### 8. Quality Assurance & Validation

#### A. Statistical Tests
- **Stationarity Tests**: Verify smoothed series properties
- **Distribution Tests**: Validate generated samples follow expected distributions
- **Correlation Analysis**: Ensure temporal relationships are preserved

#### B. Financial Validation
- **Yield Curve Shapes**: Validate generated curves are financially reasonable
- **Historical Comparison**: Compare smoothed samples with actual historical yields
- **Risk Metrics**: Calculate VaR and other risk measures for validation

### 9. Implementation Timeline

#### Week 1: Core Infrastructure
- [ ] Implement `SmoothedSampleGenerator` class
- [ ] Add chronological data loading functionality
- [ ] Create rolling statistics calculation methods

#### Week 2: Smoothing Engine
- [ ] Implement exponential moving average smoothing
- [ ] Add yield curve constraint enforcement
- [ ] Create sample generation from smoothed distributions

#### Week 3: Integration & Testing
- [ ] Integrate with existing pipeline
- [ ] Add CLI arguments and configuration
- [ ] Implement comprehensive testing suite

#### Week 4: Validation & Documentation
- [ ] Run statistical and financial validation
- [ ] Performance optimization and memory profiling
- [ ] Complete documentation and examples

### 10. Success Metrics

#### A. Technical Metrics
- **Processing Speed**: Generate smoothed samples for all time periods in <30 minutes
- **Memory Usage**: Peak memory usage <8GB for full dataset
- **File Size**: Compressed output files <50% larger than originals

#### B. Quality Metrics  
- **Statistical Validity**: Generated samples pass normality and stationarity tests
- **Financial Realism**: Smoothed yield curves maintain economically sensible relationships
- **Temporal Consistency**: Smoothed time series exhibit appropriate autocorrelation

### 11. Risk Mitigation

#### A. Data Quality
- **Missing Data**: Handle gaps in historical sample files gracefully
- **Outlier Management**: Robust smoothing methods to handle market stress periods
- **Version Control**: Maintain backward compatibility with existing pipeline

#### B. Performance Risks
- **Memory Overflow**: Implement streaming processing for large datasets  
- **Processing Time**: Parallel processing and efficient algorithms
- **Storage Space**: Configurable compression and cleanup procedures

## Conclusion

This enhancement will transform the basic percentile calculation pipeline into a sophisticated time-series smoothing and sample generation system. The proposed architecture maintains compatibility with existing infrastructure while adding powerful new capabilities for generating more realistic and temporally consistent yield curve scenarios.

The implementation prioritizes:
1. **Statistical Rigor**: Proper time-series handling and distribution modeling
2. **Financial Validity**: Maintaining economically sensible yield relationships  
3. **Computational Efficiency**: Scalable processing for large historical datasets
4. **Integration Simplicity**: Seamless addition to existing pipeline workflows

This foundation will support advanced analytics, risk modeling, and scenario generation for bond yield forecasting applications.
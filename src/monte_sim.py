import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE
from scipy.interpolate import interp1d
from KDEpy.bw_selection import silvermans_rule


class MonteCarloSimulator:
    """Monte Carlo simulator for bond yield forecasting."""

    def __init__(self, num_simulations: int = 1000, ewm_alpha: float = 0.05, use_ewm: bool = True) -> None:
        """
        Initialize the Monte Carlo simulator.

        Args:
            num_simulations: Number of Monte Carlo simulations to run
            ewm_alpha: Alpha parameter for exponential weighted mean (default: 0.1)
            use_ewm: Whether to use exponential weighted mean (default: True)
        """
        self.num_simulations = num_simulations
        self.ewm_alpha = ewm_alpha
        self.use_ewm = use_ewm
        self.xde: np.array = None
        self.ykde: np.array = None
        self.mean: float = None
        self.variance: float = None
        self.standard_dev: float = None
        self.ewm_mean: float = None
        self.ewm_variance: float = None
        self.ewm_standard_dev: float = None

    def _calculate_ewm_variance(self, y: pd.Series) -> float:
        """
        Calculate exponential weighted variance.
        
        Args:
            y: Time series data as pandas Series
            
        Returns:
            EWM variance
        """
        ewm_mean = y.ewm(alpha=self.ewm_alpha).mean().iloc[-1]
        squared_deviations = (y - ewm_mean) ** 2
        ewm_variance = squared_deviations.ewm(alpha=self.ewm_alpha).mean().iloc[-1]
        return ewm_variance

    def fit(self, y: np.array) -> None:
        """
        Fit the KDE model to the historical yield data.

        Args:
            y: Series of historical yield data
        """
        # Convert to pandas Series if numpy array
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y)
        else:
            y_series = y
            
        # Calculate traditional statistics
        self.mean = y_series.mean()
        self.variance = y_series.var()
        self.standard_dev = y_series.std()
        
        # Calculate EWM statistics if enabled
        if self.use_ewm:
            try:
                # Ensure data is sorted properly (most recent last)
                y_sorted = y_series.sort_index() if hasattr(y_series, 'index') else y_series
                
                self.ewm_mean = y_sorted.ewm(alpha=self.ewm_alpha).mean().iloc[-1]
                self.ewm_variance = self._calculate_ewm_variance(y_sorted)
                self.ewm_standard_dev = np.sqrt(self.ewm_variance)
            except Exception as e:
                # Fallback to simple statistics if EWM fails
                print(f"EWM calculation failed, falling back to simple statistics: {e}")
                self.use_ewm = False
                self.ewm_mean = self.mean
                self.ewm_variance = self.variance
                self.ewm_standard_dev = self.standard_dev
        else:
            # Use simple statistics when EWM is disabled
            self.ewm_mean = self.mean
            self.ewm_variance = self.variance
            self.ewm_standard_dev = self.standard_dev
        
        # Fit KDE using the original data
        kde = FFTKDE(bw='silverman').fit(y if isinstance(y, np.ndarray) else y.values)
        self.xde, self.ykde = kde.evaluate()

    def simulate(self, current_yield: float) -> np.array:
        """
        Run Monte Carlo simulations to generate future yield scenarios.
        Returns:
        """
        # Use EWM statistics if available, otherwise use simple statistics
        mean_to_use = self.ewm_mean if self.use_ewm and self.ewm_mean is not None else self.mean
        variance_to_use = self.ewm_variance if self.use_ewm and self.ewm_variance is not None else self.variance
        std_to_use = self.ewm_standard_dev if self.use_ewm and self.ewm_standard_dev is not None else self.standard_dev
        
        drift = mean_to_use - (0.5 * variance_to_use)

        # estimate the cdf
        dx = np.diff(self.xde)[0]
        cdf = np.cumsum(self.ykde) * dx
        cdf /= cdf[-1]


        ppf = interp1d(cdf, self.xde, bounds_error=False, fill_value=(self.xde[0], self.xde[-1]))

        # number of samples
        u = np.random.rand(1, self.num_simulations)  # uniform(0,1)
        samples = ppf(u)

        sample_changes = drift + std_to_use * samples
        
        # Handle both scalar and array current_yield inputs
        current_yield = np.atleast_1d(current_yield)
        if current_yield.ndim == 1 and len(current_yield) == 1:
            # Single yield value - broadcast to match samples
            current_yield_broadcast = current_yield[0]
            yield_predictions = current_yield_broadcast + sample_changes.flatten()
        else:
            # Multiple yield values - handle each separately


            pred = current_yield[0] + sample_changes.flatten()
            yield_predictions = np.array(pred)

        return yield_predictions

    def set_ewm_alpha(self, alpha: float) -> None:
        """
        Set the EWM alpha parameter.
        
        Args:
            alpha: Alpha parameter for exponential weighted mean
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.ewm_alpha = alpha

    def toggle_ewm(self, use_ewm: bool) -> None:
        """
        Enable or disable EWM functionality.
        
        Args:
            use_ewm: Whether to use exponential weighted mean
        """
        self.use_ewm = use_ewm

    def get_ewm_stats(self) -> dict:
        """
        Return current EWM statistics for debugging.
        
        Returns:
            Dictionary containing EWM mean, variance, and standard deviation
        """
        return {
            'ewm_mean': self.ewm_mean,
            'ewm_variance': self.ewm_variance,
            'ewm_standard_dev': self.ewm_standard_dev,
            'simple_mean': self.mean,
            'simple_variance': self.variance,
            'simple_standard_dev': self.standard_dev,
            'alpha': self.ewm_alpha,
            'use_ewm': self.use_ewm
        }

    def fit_and_simulate(self, y: np.array, current_yield: float) -> np.array:
        """
        Fit the model and run simulations.
        Args:
            y:
            current_yield:

        Returns:

        """
        self.fit(y)
        return self.simulate(current_yield)

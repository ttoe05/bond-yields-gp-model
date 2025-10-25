"""
Feature management for bond yield forecasting.
Handles loading feature configurations from YAML and feature validation.
"""

import yaml
import pandas as pd
from typing import Dict, List, Optional, Set
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureManager:
    """Manages feature selection and validation for bond yield forecasting."""
    
    def __init__(self, features_config_path: str):
        """
        Initialize the feature manager.
        
        Args:
            features_config_path: Path to the bond_important_features.yaml file
        """
        self.config_path = Path(features_config_path)
        self.features_config: Optional[Dict] = None
        # self.available_bonds: Set[str] = set()
        self._load_features_config()
    
    def _load_features_config(self) -> None:
        """Load the features configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Features config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                self.features_config = yaml.safe_load(file)
            
            self.available_time_preds = set(self.features_config.keys())
            logger.info(f"Loaded features for {len(self.available_time_preds)} future time predictions")
            logger.info(f"Available time predictions: {list(self.available_time_preds)}")
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading features config: {str(e)}")
            raise
    
    def get_features_for_time_pred(self, time_prediction: str) -> List[str]:
        """
        Get the list of features for a specific bond index.
        
        Args:
            bond_index: Name of the bond index (e.g., 'DGS10')
            
        Returns:
            List of feature column names
        """
        if self.features_config is None:
            raise ValueError("Features config must be loaded first")
        
        if time_prediction not in self.features_config:
            available_time_preds = list(self.features_config.keys())
            raise ValueError(f"Time prediction '{time_prediction}' not found in config. "
                           f"Available predictions: {available_time_preds}")
        
        features = self.features_config[time_prediction]['max_features']
        logger.info(f"Retrieved {len(features)} features for {time_prediction}")
        
        return features

    def get_dependent_variables(self):
        """
        Get the list of dependent variables for each bond index.

        Returns:
            Dict mapping bond index names to their dependent variable column names
        """
        return self.features_config['dependent_variables']

    
    def get_all_available_times(self) -> List[str]:
        """
        Get list of all bond indices with feature configurations.
        
        Returns:
            List of bond index names
        """
        return list(self.features_config.keys())
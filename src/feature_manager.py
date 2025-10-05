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
        self.available_bonds: Set[str] = set()
        self._load_features_config()
    
    def _load_features_config(self) -> None:
        """Load the features configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Features config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                self.features_config = yaml.safe_load(file)
            
            self.available_bonds = set(self.features_config.keys())
            logger.info(f"Loaded features for {len(self.available_bonds)} bond indices")
            logger.info(f"Available bonds: {list(self.available_bonds)}")
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading features config: {str(e)}")
            raise
    
    def get_features_for_bond(self, bond_index: str) -> List[str]:
        """
        Get the list of features for a specific bond index.
        
        Args:
            bond_index: Name of the bond index (e.g., 'DGS10')
            
        Returns:
            List of feature column names
        """
        if self.features_config is None:
            raise ValueError("Features config must be loaded first")
        
        if bond_index not in self.features_config:
            available_bonds = list(self.features_config.keys())
            raise ValueError(f"Bond index '{bond_index}' not found in config. "
                           f"Available bonds: {available_bonds}")
        
        features = self.features_config[bond_index]
        logger.info(f"Retrieved {len(features)} features for {bond_index}")
        
        return features
    
    # def validate_features_against_data(self, bond_index: str,
    #                                  available_columns: List[str]) -> Dict[str, any]:
    #     """
    #     Validate that required features exist in the data.
    #
    #     Args:
    #         bond_index: Name of the bond index
    #         available_columns: List of available column names in data
    #
    #     Returns:
    #         Dictionary with validation results
    #     """
    #     required_features = self.get_features_for_bond(bond_index)
    #     available_columns_set = set(available_columns)
    #     required_features_set = set(required_features)
    #
    #     validation_results = {
    #         'bond_index': bond_index,
    #         'required_features_count': len(required_features),
    #         'available_features_count': len(available_columns),
    #         'missing_features': [],
    #         'extra_features': [],
    #         'validation_passed': False
    #     }
    #
    #     # Find missing features
    #     missing_features = required_features_set - available_columns_set
    #     validation_results['missing_features'] = list(missing_features)
    #
    #     # Find extra features (available but not required)
    #     extra_features = available_columns_set - required_features_set
    #     validation_results['extra_features'] = list(extra_features)
    #
    #     # Validation passes if no features are missing
    #     validation_results['validation_passed'] = len(missing_features) == 0
    #
    #     if missing_features:
    #         logger.warning(f"Missing features for {bond_index}: {list(missing_features)}")
    #     else:
    #         logger.info(f"Feature validation passed for {bond_index}")
    #
    #     return validation_results
    #
    # def get_feature_statistics(self, bond_index: str, data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Get statistics for features of a specific bond index.
    #
    #     Args:
    #         bond_index: Name of the bond index
    #         data: DataFrame containing the features
    #
    #     Returns:
    #         DataFrame with feature statistics
    #     """
    #     features = self.get_features_for_bond(bond_index)
    #     available_features = [f for f in features if f in data.columns]
    #
    #     if len(available_features) == 0:
    #         logger.warning(f"No features found in data for {bond_index}")
    #         return pd.DataFrame()
    #
    #     feature_stats = data[available_features].describe()
    #
    #     # Add missing value counts
    #     missing_counts = data[available_features].isnull().sum()
    #     feature_stats.loc['missing_count'] = missing_counts
    #
    #     # Add data types
    #     dtypes = data[available_features].dtypes
    #     feature_stats.loc['dtype'] = dtypes
    #
    #     return feature_stats
    
    # def filter_features_by_availability(self, bond_index: str,
    #                                   available_columns: List[str]) -> List[str]:
    #     """
    #     Filter features to only include those available in the data.
    #
    #     Args:
    #         bond_index: Name of the bond index
    #         available_columns: List of available column names
    #
    #     Returns:
    #         List of available feature names
    #     """
    #     required_features = self.get_features_for_bond(bond_index)
    #     available_columns_set = set(available_columns)
    #
    #     filtered_features = [f for f in required_features if f in available_columns_set]
    #
    #     removed_count = len(required_features) - len(filtered_features)
    #     if removed_count > 0:
    #         logger.warning(f"Removed {removed_count} unavailable features for {bond_index}")
    #
    #     logger.info(f"Using {len(filtered_features)} features for {bond_index}")
    #     return filtered_features
    
    def get_all_available_bonds(self) -> List[str]:
        """
        Get list of all bond indices with feature configurations.
        
        Returns:
            List of bond index names
        """
        return list(self.available_bonds)
    
    def create_feature_importance_template(self, bond_index: str, 
                                         feature_importances: Dict[str, float]) -> Dict:
        """
        Create a template for storing feature importances.
        
        Args:
            bond_index: Name of the bond index
            feature_importances: Dictionary mapping feature names to importance scores
            
        Returns:
            Dictionary with structured feature importance data
        """
        features = self.get_features_for_bond(bond_index)
        
        template = {
            'bond_index': bond_index,
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_features': len(features),
            'feature_importances': {}
        }
        
        for feature in features:
            importance = feature_importances.get(feature, 0.0)
            template['feature_importances'][feature] = {
                'importance_score': importance,
                'rank': 0  # Will be filled by ranking function
            }
        
        # Add rankings
        sorted_features = sorted(template['feature_importances'].items(), 
                               key=lambda x: x[1]['importance_score'], reverse=True)
        
        for rank, (feature, data) in enumerate(sorted_features, 1):
            template['feature_importances'][feature]['rank'] = rank
        
        return template
    
    def export_feature_config(self, output_path: str, bond_configs: Dict[str, List[str]]) -> None:
        """
        Export feature configuration to YAML file.
        
        Args:
            output_path: Path for output YAML file
            bond_configs: Dictionary mapping bond indices to feature lists
        """
        try:
            with open(output_path, 'w') as file:
                yaml.dump(bond_configs, file, default_flow_style=False, sort_keys=True)
            
            logger.info(f"Feature configuration exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting feature config: {str(e)}")
            raise
    
    # def validate_feature_types(self, bond_index: str, data: pd.DataFrame) -> Dict[str, any]:
    #     """
    #     Validate that features have appropriate data types.
    #
    #     Args:
    #         bond_index: Name of the bond index
    #         data: DataFrame containing the features
    #
    #     Returns:
    #         Dictionary with type validation results
    #     """
    #     features = self.get_features_for_bond(bond_index)
    #     available_features = [f for f in features if f in data.columns]
    #
    #     validation_results = {
    #         'bond_index': bond_index,
    #         'numeric_features': [],
    #         'non_numeric_features': [],
    #         'infinite_values': [],
    #         'validation_passed': True
    #     }
    #
    #     for feature in available_features:
    #         series = data[feature]
    #
    #         if pd.api.types.is_numeric_dtype(series):
    #             validation_results['numeric_features'].append(feature)
    #
    #             # Check for infinite values
    #             if np.isinf(series).any():
    #                 validation_results['infinite_values'].append(feature)
    #                 validation_results['validation_passed'] = False
    #         else:
    #             validation_results['non_numeric_features'].append(feature)
    #             validation_results['validation_passed'] = False
    #
    #     if validation_results['non_numeric_features']:
    #         logger.warning(f"Non-numeric features found: {validation_results['non_numeric_features']}")
    #
    #     if validation_results['infinite_values']:
    #         logger.warning(f"Features with infinite values: {validation_results['infinite_values']}")
    #
    #     return validation_results
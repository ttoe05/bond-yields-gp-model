import pandas as pd
from data_loader import BondDataLoader
from feature_manager_sim import FeatureManagerSim
from feature_manager import FeatureManager
from kernel_ridge_models import KernelRidgeEnsemble
from walk_forward import WalkForwardValidator
from random import uniform
# import floor
from math import floor
# import time
import time

if __name__ == "__main__":
    time_prediction_list = ["thirty-day-ahead", "sixty-day-ahead"]
    num_prediction_list = [30, 60]
    for time_prediction, num in zip(time_prediction_list, num_prediction_list):
        # get the target, actual, and feature columns
        feature_manager = FeatureManagerSim(features_config_path='data/features_selected4.yaml')
        feature_manager2 = FeatureManager(features_config_path='data/features_selected3.yaml')
        independent_variables = feature_manager2.get_features_for_time_pred(time_prediction=time_prediction)
        dependent_variables = feature_manager.get_dependent_variables(time_prediction=time_prediction)
        actuals_variables = feature_manager.get_actual_variables()

        # print the outputs
        print(f"Independent variables: {independent_variables}")
        dependent_variable_new = [f"{x}_future_val" for x in dependent_variables]
        print(f"Dependent variables: {dependent_variable_new}")
        print(f"Actual variables: {actuals_variables}")

        # raw_data = pd.read_parquet(f'data/kernel_regression_train_shifted_{num}_.parquet')
        # raw_data[feature_manager2.get_dependent_variables()].head()

        data_loader = BondDataLoader(data_path=f'data/kernel_regression_train_shifted_{num}_.parquet')
        independent_variables = feature_manager2.get_features_for_time_pred(time_prediction=time_prediction)
        dependent_variables = feature_manager2.get_dependent_variables()
        print(dependent_variables)
        data_loader.load_data(x=independent_variables, y=dependent_variables, actuals=dependent_variables)
        # print(data_loader.data[dependent_variables].head())
        krr_ensemble = KernelRidgeEnsemble(training_metric='r2_flat', random_state=42)
        walk_forward = WalkForwardValidator(
            model=krr_ensemble,
            data_loader=data_loader,
            feature_manager=feature_manager2,
            time_prediction=time_prediction,
            window_size=2000,
            min_window_size=2000,
            step_size=1,
            model_retrain_interval=31,
            n_parallel_jobs=4,
            use_scaling=True,
            window_type='growing'
        )

        walk_forward.run_walk_forward_validation()
        walk_forward.export_results(filepath=f'results/{time_prediction}')
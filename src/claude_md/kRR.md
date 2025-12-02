Kernel Ridge Regression

Devise a plan to create a kernel ridge regression modeling Experiments class object that outputs future index yields using a set of macro economic variables. The experiments object should include the following 


Initialization:
* Passed Parameters to the object:
    * Training metric: the selected metric for choosing the best model parameters during grid search time series cv of the kernel ridge regression model available metrics to select from are ‘mse’, ‘mse_flat, ‘rmse', ‘r2_avg', 'r2_flat’. This parameter accepts string values
    * Random State: the random state for randomization and reproducibility
    * N jobs: the number of jobs to use during the grid search time series cv
* Attributes create upon initialization
    * Best model equals None
    * Kernels a list of available kernels to be passed to the grid search time cv when training
Methods:
* Create kernel configuration
    * Returns a list of available kernels. The kernels to use are the 5 kernels explored in the notebook analysis/kernel_regression.ipynb
    * Set the list to the attribute kernels
* Train KRR
    * Training of the kernel ridge regression using grid search time series cross validation. Leverage the code for training the model in the notebook analysis/kernel_regression.ipynb
    * Take the following parameters:
        * x_train: the feature variables used for training
        * y_train: the multivariate dependent variable for predicting.
        * alpa_params: list of alpha parameters to test against each other. Default to the alpha that is used in analysis/kernel_regression.ipynb
    * Return the best estimator to the best model attribute
    * Set the attribute residuals to the residuals of the training set predictions of the multivariate output
    * Return training metrics as a dictionary
        * Return the available performance metrics, the best kernel parameter, the best alpha parameter
* Create the functions for calculating the metrics ‘mse’, ‘mse_flat, ‘rmse', ‘r2_avg', 'r2_flat’. The output of predictions will be from a MultiOutputRegressor object.
* Create a point prediction wrapper function that predicts a given input using the best model estimator.
* Sample predictions function
    * Get the point prediction from the prediction variable
    * Get the mean absolute average of the residuals
    * Calculate the covariance matrix of the residuals 
    * Use the multivariate normal distribution function to return a distribution of the predictions to the right and left of the expected mean prediction of the KRR model. Set the mean as the mean point prediction, the standard deviation as the absolute average of residuals and the covariance to the calculated covariance of the residuals
    * Out the samples as a pandas data frame with the dependent variable as the columns. Generate 1000 samples


Refactor Walk forward

Make the following refactors to walk forward
* Add standard scaling of the feature values and the dependent variables before being passed to train the models and for predictions
* Add inverse transformation of point predictions and samples that are being persisted.


Refactor Data Loader

Add the ability to test with a growing window
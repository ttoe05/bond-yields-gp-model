GP Modeling 


Devise a plan to create a GP modeling Experiments class object. The experiments object should include the following and will be used for forecasting future changes in index yields.

Initialization:
- Inputs:
    - Raw data for analysis, training of the model, and forecasting new values
    - A List of the columns to use for predictions
    - A list of columns to identify the feature variables
Methods:
- Feature transformation
    - Inputs:
        - The function takes in a data frame that is the subset of features that will be transformed. Call the input parameter x_features
    - Function:
        - Use standardScaler to normalize the features
        - Return the fitted scaler object and the newly transformed features
- Train GP Model
    - Wrapper function to GaussianProcessRegressor.
    - Inputs:
        - The x_features data frame.
        - The y_features pandas series
        - Kernel a defined kernel
    - Function
        - Run fit transform on the features and dependent variable, y_features
        - Predict the training samples and get the residuals 
        - Set the class attribute model to the trained GP model
        - Output metrics on the trained model as a dictionary
            - Kernel
            - log_marginal_likelihood_value_float
            - Residuals of the training sample
- Predict:
    - Wrapper function to  GaussianProcessRegressor function predict
    - Input
        - The x_feature variables used for making predictions
    - Function
        - Call the class variable model’s predict function using the passed feature variables. Set the function values return_std, and return_cov equal to True
        - Function returns the prediction values, the std_deviation, and covariance as a dictionary
- Predict samples:
    - Input
        - The x_feature variables used for making predictions
    - Function
        - Call the fitted model’s function y_samples with 1000 n samples
        - Convert the nd array of size n_samples, x_features to a pandas data frame
        - Return the data frame

- Fit transform predict pipeline
    - Inputs
        - X-train: data frame of x features for training
        - Y-train pd.Series for y features for training
        - Kernel: kernel to use for the gpr model
        - x_predict: data frame of x features for prediction
        - Samples defaulted to True
    - Function
        - Transform the features using the feature transformation function and save the output standardScaler to a variable named normalizer
        - Fit the GP model using the train gp model function save output metrics to the variable training metrics
        - Transform the x_predict features using the normalizer variable and save to a variable called x_predict_normalized
        - Pass the x_predict_normalized features to the predict function and save the out put to the variable predict_metrics
        - If samples is set to true run the predict samples function and save the function to the variable y_samples else set y_samples variable to None
        - Return a tuple of the training metrics, predict_metrics, and y_samples

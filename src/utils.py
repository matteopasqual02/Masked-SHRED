import numpy as np  #type: ignore
import torch        #type: ignore

def get_train_medians(Xtrain):
    # compute the median of each sensor using numpy
    # input shape is (n_samples, n_sensors) or (n_samples, n_timestamps, n_sensors)
    
    X_reshaped = Xtrain.reshape(-1, Xtrain.shape[-1])       # reshape to (n_samples * n_timestamps, n_sensors) if 3D
    medians = np.nanmedian(X_reshaped, axis=0)              # compute median along the first dimension, ignoring NaNs
    medians = np.nan_to_num(medians, nan=0.0)               # replace NaNs with 0.0

    return medians

def get_train_means(Xtrain):
    # compute the mean of each sensor using numpy
    # input shape is (n_samples, n_sensors) or (n_samples, n_timestamps, n_sensors)
    
    X_reshaped = Xtrain.reshape(-1, Xtrain.shape[-1])      # reshape to (n_samples * n_timestamps, n_sensors) if 3D
    means = np.nanmean(X_reshaped, axis=0)                 # compute mean along the first dimension, ignoring NaNs
    means = np.nan_to_num(means, nan=0.0)                  # replace NaNs with 0.0

    return means

def apply_imputation(X, medians):
    # perform median imputation on the input data X using the provided medians
    # (on a copy of X to avoid modifying the original data)
    
    X_imputed = X.copy()                                    # create a copy of X to avoid modifying the original data
    nan_indices = np.isnan(X_imputed)                       # find indices of NaN values
    
    # Replace NaN values with corresponding medians for each sensor (column)
    for sensor_idx in range(X_imputed.shape[1]):
        X_imputed[nan_indices[:, sensor_idx], sensor_idx] = medians[sensor_idx]

    return X_imputed

def median_imputer_single_dataset(X, train_idx):
    # compute medians from the training data and apply median imputation to the entire dataset (train, val, test)

    Xtrain = X[train_idx]                       # extract training data using train indices
    medians = get_train_medians(Xtrain)         # compute medians from the training data
    X_imputed = apply_imputation(X, medians)    # apply median imputation to the entire dataset

    return X_imputed

def mean_imputer_single_dataset(X, train_idx):
    # compute means from the training data and apply mean imputation to the entire dataset (train, val, test)

    Xtrain = X[train_idx]                       # extract training data using train indices
    means = get_train_means(Xtrain)             # compute means from the training data
    X_imputed = apply_imputation(X, means)      # apply mean imputation to the entire dataset

    return X_imputed

def last_value_imputer_single_dataset(X, train_idx):
    # perform last value imputation on the input data X using the last observed value 
    
    X_imputed = X.copy()                                    # create a copy of X to avoid modifying the original data
    for sensor_idx in range(X_imputed.shape[1]):
        ts = X_imputed[:, sensor_idx]                       # extract time series for the current sensor
        nan_indices = np.where(np.isnan(ts))[0]             # find indices of NaN
        if len(nan_indices) == 0:
            continue
        for idx in nan_indices:
            ts[idx] = 0.0 if idx == 0 else ts[idx - 1]      # replace NaN with last observed value (or 0.0 if it's the first value)

    return X_imputed
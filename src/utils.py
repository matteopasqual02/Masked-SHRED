import numpy as np  #type: ignore
import matplotlib.pyplot as plt  #type: ignore
import torch  #type: ignore
from torch.utils.data import DataLoader  #type: ignore

## This file contains utility functions for data imputation and preprocessing, including:
# - get_train_medians: compute the median of each sensor from the training data
# - get_train_means: compute the mean of each sensor from the training data
# - apply_imputation: perform median or mean imputation on the input data using the provided            
#   medians or means
# - median_imputer_single_dataset: compute medians from the training data and apply median imputation to the entire dataset (train, val, test)
# - mean_imputer_single_dataset: compute means from the training data and apply mean imputation to the entire dataset (train, val, test)
# - last_value_imputer_single_dataset: perform last value imputation on the input data using the last observed value 
# - generate_GRUD_4D_input: generate a 4D input tensor for GRUD from the original measurements, including the original values, last observed values, masks, and time since last observation
# - generate_channels: generate interleaved data and mask channels for each sensor from the original measurements   

def _to_numpy_array(X):
    # Convert tensor-like inputs to numpy arrays for numpy-based imputation routines.
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "detach") and hasattr(X, "cpu") and hasattr(X, "numpy"):
        return X.detach().cpu().numpy()
    return np.asarray(X)

def get_train_medians(Xtrain):
    # compute the median of each sensor using numpy
    # input shape is (n_samples, n_sensors) or (n_samples, n_timestamps, n_sensors)
    
    X_reshaped = Xtrain.reshape(-1, Xtrain.shape[-1])       # reshape to (n_samples * n_timestamps, n_sensors) if 3D
    medians = np.nanmedian(X_reshaped, axis=0)              # compute median along the first dimension, ignoring NaNs
    medians = np.nan_to_num(medians, nan=0.0)               # replace NaNs with 0.0

    return medians

def get_train_medians_3D(X_train):
    
    # Calculate median across Batch (axis 0) and Time (axis 1) dimensions
    # to obtain a specific median value for each individual sensor.
    # Reshape to combine batch and time dimensions, then compute median
    X_reshaped = X_train.reshape(-1, X_train.shape[2])  # Shape: (batch*time, sensors)
    medians = np.nanmedian(X_reshaped, axis=0)
    
    # Handle edge case: if a sensor is entirely NaN in the training set,
    # fill its median with 0.0 to prevent propagating NaNs during transform.
    medians = np.nan_to_num(medians, nan=0.0)

    return medians
    
def get_train_means(Xtrain):
    # compute the mean of each sensor using numpy
    # input shape is (n_samples, n_sensors) or (n_samples, n_timestamps, n_sensors)
    
    X_reshaped = Xtrain.reshape(-1, Xtrain.shape[-1])      # reshape to (n_samples * n_timestamps, n_sensors) if 3D
    means = np.nanmean(X_reshaped, axis=0)                 # compute mean along the first dimension, ignoring NaNs
    means = np.nan_to_num(means, nan=0.0)                  # replace NaNs with 0.0

    return means

def get_train_means_3D(X_train):
    
    # Calculate mean across Batch (axis 0) and Time (axis 1) dimensions
    # to obtain a specific mean value for each individual sensor.
    # Reshape to combine batch and time dimensions, then compute mean
    X_reshaped = X_train.reshape(-1, X_train.shape[2])  # Shape: (batch*time, sensors)
    means = np.nanmean(X_reshaped, axis=0)           # Compute mean while ignoring NaNs
    
    # Handle edge case: if a sensor is entirely NaN in the training set,
    # fill its mean with 0.0 to prevent propagating NaNs during transform.
    means = np.nan_to_num(means, nan=0.0)

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

def apply_imputation_3D(X, medians):
    # Create a copy to prevent overwriting the original input array
    X_imputed = X.copy()

    # Locate the indices where values are NaN
    nan_indices = np.where(np.isnan(X_imputed))

    # Replace NaNs by mapping the sensor index (nan_indices[2])
    # to its corresponding pre-calculated statistic value.
    X_imputed[nan_indices] = medians[nan_indices[2]]
    return X_imputed

def median_imputer_single_dataset(X, train_idx):
    # compute medians from the training data and apply median imputation to the entire dataset (train, val, test)

    X = _to_numpy_array(X)

    if X.ndim == 2:
        Xtrain = X[train_idx]                       # extract training data using train indices
        medians = get_train_medians(Xtrain)         # compute medians from the training data
        X_imputed = apply_imputation(X, medians)    # apply median imputation to the entire dataset
    elif X.ndim == 3:
        Xtrain = X[train_idx, :, :]                 # extract training data using train indices
        medians = get_train_medians_3D(Xtrain)      # compute medians from the training data
        X_imputed = apply_imputation_3D(X, medians) # apply median imputation to the entire dataset
    else:
        raise ValueError("Input X must be a 2D or 3D array")
    
    return X_imputed

def mean_imputer_single_dataset(X, train_idx):
    # compute means from the training data and apply mean imputation to the entire dataset (train, val, test)

    X = _to_numpy_array(X)

    if X.ndim == 2:
        Xtrain = X[train_idx]                       # extract training data using train indices
        means = get_train_means(Xtrain)             # compute means from the training data
        X_imputed = apply_imputation(X, means)      # apply mean imputation to the entire dataset
    elif X.ndim == 3:
        Xtrain = X[train_idx, :, :]                 # extract training data using train indices
        means = get_train_means_3D(Xtrain)          # compute means from the training data
        X_imputed = apply_imputation_3D(X, means)   # apply mean imputation to the entire dataset
    else:
        raise ValueError("Input X must be a 2D or 3D array")
    
    return X_imputed

def last_value_imputer_single_dataset(X, train_idx):
    # perform last value imputation on the input data X using the last observed value 

    X = _to_numpy_array(X)
    X_imputed = X.copy()                                    # create a copy of X to avoid modifying the original data
    if X_imputed.ndim == 2:
        for sensor_idx in range(X_imputed.shape[1]):
            ts = X_imputed[:, sensor_idx]                       # extract time series for the current sensor
            nan_indices = np.where(np.isnan(ts))[0]             # find indices of NaN
            if len(nan_indices) == 0:
                continue
            for idx in nan_indices:
                ts[idx] = 0.0 if idx == 0 else ts[idx - 1]      # replace NaN with last observed value (or 0.0 if it's the first value)
    elif X_imputed.ndim == 3:
        for sample_idx in range(X_imputed.shape[0]):
            for sensor_idx in range(X_imputed.shape[2]):
                ts = X_imputed[sample_idx, :, sensor_idx]        # time series for one sample and one sensor
                nan_indices = np.where(np.isnan(ts))[0]          # find indices of NaN in this sequence
                if len(nan_indices) == 0:
                    continue
                for idx in nan_indices:
                    ts[idx] = 0.0 if idx == 0 else ts[idx - 1]   # forward-fill along time for each sample/sensor
    else:
        raise ValueError("Input X must be a 2D or 3D array")

    return X_imputed

def generate_GRUD_4D_input(measurements):
    # generate a 4D input tensor for GRUD from the original measurements
    # input shape: (T, n_sensors)
    # output shape: (T, 4, n_sensors) where the 4 channels are [X, X_last_obsv, Mask, Delta]

    measurements = np.asarray(measurements)
    T, n_sensors = measurements.shape
    Mask = (~np.isnan(measurements)).astype(float)                  # Mask indicating observed values (1.0 for observed, 0.0 for missing)
    X = np.nan_to_num(measurements, nan=0.0).astype(np.float32)     # Original measurements with NaNs replaced by 0.0

    X_last_obsv = np.zeros_like(X)                                  # To store the last observed value for each sensor
    Delta = np.zeros_like(X)                                        # To store the time since last observation for each sensor

    # Iterate through each sensor 
    for sensor_idx in range(n_sensors):
        last_value = 0.0
        time_since_last_obs = 0.0

        # Iterate through time steps for the current sensor
        for t in range(T):
            if Mask[t, sensor_idx] == 1.0:          # If the value is observed
                last_value = X[t, sensor_idx]       # Update last observed value
                time_since_last_obs = 0.0           # Reset time since last observation
            else:
                time_since_last_obs += 1.0          # Increment time since last observation

            X_last_obsv[t, sensor_idx] = last_value         # Store the last observed value
            Delta[t, sensor_idx] = time_since_last_obs      # Store the time since last observation

    # stack the channels to create a 4D tensor of shape (4, T, n_sensors)
    GRUD_input = np.stack((X, X_last_obsv, Mask, Delta), axis=0)

    return GRUD_input

def generate_GRUD_4D_input_3D(measurements):
    # generate a 3D input tensor for GRUD from the original measurements
    # input shape: (n_samples, T, n_sensors)
    # output shape: (n_samples, T, 4, n_sensors) where the 4 channels are [X, X_last_obsv, Mask, Delta]

    n_samples, T, n_sensors = measurements.shape
    Mask = (~np.isnan(measurements)).astype(float)                  # Mask indicating observed values (1.0 for observed, 0.0 for missing)
    X = np.nan_to_num(measurements, nan=0.0).astype(np.float32)     # Original measurements with NaNs replaced by 0.0

    X_last_obsv = np.zeros_like(X)                                  # To store the last observed value for each sensor
    Delta = np.zeros_like(X)                                        # To store the time since last observation for each sensor

    # Iterate through each sample and sensor 
    for sample_idx in range(n_samples):
        for sensor_idx in range(n_sensors):
            last_value = 0.0
            time_since_last_obs = 0.0

            # Iterate through time steps for the current sample/sensor
            for t in range(T):
                if Mask[sample_idx, t, sensor_idx] == 1.0:          # If the value is observed
                    last_value = X[sample_idx, t, sensor_idx]       # Update last observed value
                    time_since_last_obs = 0.0                       # Reset time since last observation
                else:
                    time_since_last_obs += 1.0                      # Increment time since last observation

                X_last_obsv[sample_idx, t, sensor_idx] = last_value         # Store the last observed value
                Delta[sample_idx, t, sensor_idx] = time_since_last_obs      # Store the time since last observation

    # stack the channels to create a 4D tensor of shape (n_samples, T, 4, n_sensors)
    GRUD_input = np.stack((X, X_last_obsv, Mask, Delta), axis=2)

    return GRUD_input

def generate_channels(measurements):

    # the input is a matrix (ndata, nsensor) or (ndata, lags, nsenso)
    # i want to generate (ndata, nsensor*2) or (ndata, lags, nsensor*2) 
    # the first half of the channels are the original measurements with NaNs replaced by 0.0
    # the second half are binary masks indicating observed values (1.0 for observed, 0.0 for missing)

    mask = (~np.isnan(measurements)).astype(float)                              # Mask indicating observed values (1.0 for observed, 0.0 for missing)
    measurements = np.nan_to_num(measurements, nan=0.0).astype(np.float32)      # Original measurements with NaNs replaced by 0.0

    # fill data1 - mask1 - data2 - mask2 - ... for each sensor
    if measurements.ndim == 2:  # (ndata, nsensor)
        ndata, nsensor = measurements.shape
        channels = np.zeros((ndata, nsensor * 2), dtype=np.float32)
        for i in range(nsensor):
            channels[:, 2*i] = measurements[:, i]   # data channel
            channels[:, 2*i + 1] = mask[:, i]       # mask channel
    elif measurements.ndim == 3:  # (ndata, lags, nsensor)
        ndata, lags, nsensor = measurements.shape
        channels = np.zeros((ndata, lags, nsensor * 2), dtype=np.float32)
        for i in range(nsensor):
            channels[:, :, 2*i] = measurements[:, :, i]   # data channel
            channels[:, :, 2*i + 1] = mask[:, :, i]       # mask channel
    else:
        raise ValueError("Input measurements must be 2D or 3D array")
    
    #return a tensor of shape (ndata, nsensor*2) or (ndata, lags, nsensor*2) where the channels are interleaved as [data1, mask1, data2, mask2, ...]
    return channels

## This file also contains a utility function for plotting training curves: 
# - plot_training_curves: plot the training and validation MSE loss curves over epochs using matplotlib

def plot_training_curves(train_errors, valid_errors):

    # PLOT TRAINING CURVES
    plt.figure(figsize=(10, 5))
    plt.plot(train_errors, label='Train MSE')
    plt.plot(valid_errors, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation MSE Loss Curves')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_reconstruction_metrics(reconstructions, ground_truths):
    """Compute MSE, MRE, MAE, and NMAE for each split and reconstructed field.

    Parameters
    ----------
    reconstructions : Mapping
        Mapping of split name to a mapping of field name to predictions.
    ground_truths : Mapping
        Mapping with the same split and field names as ``reconstructions``.

    Returns
    -------
    dict
        ``metrics[split][field]`` contains ``mse``, ``mre``, ``mae``, and
        ``nmae`` as Python floats. Inputs may be NumPy arrays or torch tensors.
    """
    if set(reconstructions) != set(ground_truths):
        raise ValueError("reconstructions and ground_truths must contain the same splits")

    metrics = {}
    for split, split_reconstructions in reconstructions.items():
        split_ground_truths = ground_truths[split]
        if set(split_reconstructions) != set(split_ground_truths):
            raise ValueError(
                f"{split}: reconstructions and ground_truths must contain the same fields"
            )

        split_metrics = {}
        for field, prediction in split_reconstructions.items():
            predicted_tensor = torch.as_tensor(prediction, dtype=torch.float64)
            truth_tensor = torch.as_tensor(split_ground_truths[field], dtype=torch.float64)
            if predicted_tensor.shape != truth_tensor.shape:
                raise ValueError(
                    f"{split}/{field}: prediction shape {tuple(predicted_tensor.shape)} "
                    f"does not match ground-truth shape {tuple(truth_tensor.shape)}"
                )

            error = predicted_tensor - truth_tensor
            mse_value = (error ** 2).sum(axis=-1).mean() / truth_tensor.shape[-1]
            mre_value = (
                (truth_tensor - predicted_tensor).pow(2).sum(axis=-1).sqrt()
                / truth_tensor.pow(2).sum(axis=-1).sqrt()
            ).mean()
            denominator = torch.amax(truth_tensor) - torch.amin(truth_tensor)
            split_metrics[field] = {
                "mse": mse_value.item(),
                "mre": mre_value.item(),
                "mae": torch.mean(torch.abs(error)).item(),
                "nmae": torch.mean(torch.abs(error) / denominator).item(),
            }
        metrics[split] = split_metrics
    del reconstructions, ground_truths
    return metrics

def evaluate_model(
    evaluate,
    mshred,
    train_dataset,
    val_dataset,
    test_dataset,
    model,
    type,
    masking_type=None,
):
    def _compute_nrmse(dataset):
        mse_loss = torch.nn.MSELoss(reduction="sum")
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        total_loss = 0.0
        total_target_energy = 0.0
        is_mask_data = masking_type == "mask_data"

        mshred.eval()
        with torch.no_grad():
            if is_mask_data:
                for X, M, Y in dataloader:
                    outputs = mshred(X, M)
                    total_loss += mse_loss(outputs, Y).item()
                    total_target_energy += Y.pow(2).sum().item()
            else:
                for X, Y in dataloader:
                    outputs = mshred(X)
                    total_loss += mse_loss(outputs, Y).item()
                    total_target_energy += Y.pow(2).sum().item()

        return ((total_loss / max(total_target_energy, 1e-12)) ** 0.5) / 2

    def _evaluate_with_optional_nrmse(dataset):
        results = evaluate(mshred, dataset, masking_type=masking_type) if masking_type is not None else evaluate(mshred, dataset)
        if len(results) == 3:
            return results
        if hasattr(results, "nrmse"):
            mse, mre = results
            return mse, mre, float(results.nrmse)
        if len(results) == 2:
            mse, mre = results
            return mse, mre, _compute_nrmse(dataset)
        raise ValueError(f"Expected evaluate() to return 2 or 3 values, got {len(results)}")
    
    # MSE and MRE
    if masking_type is None:
        train_mse, train_mre, train_nrmse = _evaluate_with_optional_nrmse(train_dataset)
        val_mse, val_mre, val_nrmse = _evaluate_with_optional_nrmse(val_dataset)
        test_mse, test_mre, test_nrmse = _evaluate_with_optional_nrmse(test_dataset)
    else:
        train_mse, train_mre, train_nrmse = _evaluate_with_optional_nrmse(train_dataset)
        val_mse, val_mre, val_nrmse = _evaluate_with_optional_nrmse(val_dataset)
        test_mse, test_mre, test_nrmse = _evaluate_with_optional_nrmse(test_dataset)
        
    test_error = {
        "model": model,
        "type": type,
        "train-mse": float(train_mse),
        "val-mse": float(val_mse),
        "test-mse": float(test_mse),
        "train-mre": float(train_mre),
        "val-mre": float(val_mre),
        "test-mre": float(test_mre),
        "train-nrmse": float(train_nrmse),
        "val-nrmse": float(val_nrmse),
        "test-nrmse": float(test_nrmse),
    }
    return test_error
from typing import Union, List, Tuple, Optional     #type: ignore
import numpy as np                                  #type: ignore
import torch                                        #type: ignore  
import pandas as pd                                 #type: ignore
from itertools import count                         #type: ignore   
from sklearn.preprocessing import MinMaxScaler, StandardScaler  #type: ignore
from sklearn.utils.extmath import randomized_svd                #type: ignore
import matplotlib.pyplot as plt                     #type: ignore

from .utils import median_imputer_single_dataset, mean_imputer_single_dataset, last_value_imputer_single_dataset
from .utils import generate_GRUD_4D_input, generate_channels

DEFAULT_MODES = 50
DataInput = Union[
    str,                # .npy/.npz file path
    np.ndarray,         # raw NumPy
    torch.Tensor,       # PyTorch tensor
]

# data manager class
class DataManager:

    def __init__(
        self,
        lags: int = 20,             # number of past time steps to use as input
        train_size: float = 0.8,    # percentage of data to use for training
        val_size: float = 0.1,      # percentage of data to use for validation
        test_size: float = 0.1,     # percentage of data to use for testing
    ):
        
        #check that the sizes sum to 1
        if not abs(train_size + val_size + test_size - 1.0) < 1e-8:
            raise ValueError("train_size, val_size and test_size must sum to 1")
        
        # store the parameters
        self.lags = lags
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # other params
        self.sensor_summary_df = None
        self.sensor_measurements_df = None
        self.sensor_measurements = None
        self.data = None
        self.sensor_scaler = None
        self.data_scaler = None

        self._dataset_ids = []
        self._active_dataset_ids = []
        self._dataset_spatial_shape = {}
        self._dataset_lengths = {}              # spatial dim after flattening to a single spatial axis
        self._Vt_registry = {}
        self._preSVD_scaler_registry = {}
        self._sensor_number = count(start=0)

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.train_sensor_measurements = None
        self.val_sensor_measurements = None
        self.test_sensor_measurements = None

    def add_data(
        self,
        data: DataInput,
        id: str,
        random: Optional[int] = None,                                   # number of random sensors to select (if None, use all sensors)
        stationary: Optional[Union[Tuple, List[Tuple]]] = None,         # coordinates of stationary sensors (if None, use random sensors)
        mobile: Optional[Union[List[Tuple], List[List[Tuple]]]] = None, # coordinates of mobile sensors (if None, use random sensors)
        measurements: Optional[np.ndarray] = None,                      # sensor measurements (if None, will be loaded from data)
        compress: Union[None, bool, int] = True,                        # whether to apply SVD compression to the data (if True, compress to x components; if None or False, do not compress)
        seed: Optional[int] = None,                                     # random seed for reproducibility (if None, no seed is set)
        imputer: Optional[str] = None,                                  # imputation strategy to apply to the provided measurements (if None, no imputation is applied; if 'median', median imputation is applied using the training data)
        verbose: bool = False
    ):
        
        # check that the id is unique
        if id in self._dataset_ids:
            raise ValueError(f"Dataset with id '{id}' already exists. Please choose a unique id for each dataset.")
        
        modes = parse_compress(compress) 
        data = get_data(data)
        dataset_spatial_shape = data.shape[1:]  # assume data is (time, spatial_dims...)

        #indices for train/val/test split
        self.train_indices = self.train_indices if self.train_indices is not None else np.arange(0, int(len(data) * self.train_size))
        self.val_indices = self.val_indices if self.val_indices is not None else np.arange(int(len(data) * self.train_size), int(len(data) * (self.train_size + self.val_size)))
        self.test_indices = self.test_indices if self.test_indices is not None else np.arange(int(len(data) * (self.train_size + self.val_size)), len(data))

        # sensor dict
        sensors_dict = get_sensor_measurements(
            data, id, self._sensor_number, random, stationary, mobile, measurements, seed,
            train_indices = self.train_indices, imputer=imputer
        )
        new_sensor_summary_df = sensors_dict['sensor_summary']
        new_sensor_measurements_df = sensors_dict['sensor_measurements_df']
        new_sensor_measurements = sensors_dict['sensor_measurements']

        # dimensionality reduction
        data = data.reshape(data.shape[0], -1)  # flatten spatial dimensions to a single axis
        if modes > 0:
            sc = StandardScaler()
            sc.fit(data[self.train_indices])
            data = sc.transform(data)
            U, S, Vt = randomized_svd(data[self.train_indices], n_components=modes, n_iter='auto')

            #plot singular values decay
            reconstruction = U @ np.diag(S) @ Vt
            residual_energy = np.linalg.norm(data[self.train_indices] - reconstruction) ** 2 \
                            / np.linalg.norm(data[self.train_indices]) ** 2

            data = data @ Vt.T

            if verbose:
                plt.semilogy(S[:modes], color = 'magenta', linewidth = 2)
                plt.title("Singular Values Decay \n")
                plt.show()
            
            print(f"Residual energy of {id} after keeping {modes} modes: {residual_energy*100:.4f} %")

        # commit to self
        self._dataset_ids.append(id)
        self._active_dataset_ids = list(self._dataset_ids)
        self._dataset_spatial_shape[self._dataset_ids[-1]] = dataset_spatial_shape
        self.train_indices = self.train_indices
        self.val_indices = self.val_indices
        self.test_indices = self.test_indices
        if self.sensor_measurements_df is None:
            self.sensor_measurements_df = new_sensor_measurements_df
        else:
            self.sensor_measurements_df = pd.concat([self.sensor_measurements_df, new_sensor_measurements_df], axis=1)
        if self.sensor_summary_df is None:
            self.sensor_summary_df = new_sensor_summary_df
        else:
            self.sensor_summary_df = pd.concat([self.sensor_summary_df, new_sensor_summary_df], axis=0).reset_index(drop=True)
        if new_sensor_measurements is not None:
            if self.sensor_measurements is None:
                self.sensor_measurements = new_sensor_measurements
            else:
                self.sensor_measurements = np.hstack((self.sensor_measurements, new_sensor_measurements))
            self.train_sensor_measurements = self.sensor_measurements[self.train_indices]
            self.val_sensor_measurements = self.sensor_measurements[self.val_indices]
            self.test_sensor_measurements = self.sensor_measurements[self.test_indices]
        if modes > 0:
            self._preSVD_scaler_registry[self._dataset_ids[-1]] = sc
            self._Vt_registry[self._dataset_ids[-1]] = Vt
        self._dataset_lengths[self._dataset_ids[-1]] = data.shape[1]  # save spatial dim to registry
        if self.data is None:
            self.data = data
        else:
            self.data = np.hstack((self.data, data))

    def prepare(self, verbose = False, masking_type = None, unused_data = None):

        if self.data is None:
            raise ValueError("No data found. Please add at least one dataset before calling prepare().")

        print(f"\nDatasets: {self._dataset_ids}")
        if unused_data is not None:
            excluded_ids = {unused_data} if isinstance(unused_data, str) else set(unused_data)
            print(f"Dataset NOT used: {sorted(excluded_ids)}")
            self._active_dataset_ids = [ids for ids in self._dataset_ids if ids not in excluded_ids]
        else:
            self._active_dataset_ids = list(self._dataset_ids)

        if not self._active_dataset_ids:
            raise ValueError("All datasets were excluded via unused_data; no output targets remain.")

        data_blocks = []
        col_start = 0
        for ids in self._dataset_ids:
            col_end = col_start + self._dataset_lengths[ids]
            if ids in self._active_dataset_ids:
                data_blocks.append(self.data[:, col_start:col_end])
            col_start = col_end

        data = np.hstack(data_blocks)
            
        sc = MinMaxScaler()
        sc.fit(data[self.train_indices])
        self.data_scaler = sc
        scaled_data = sc.transform(data)

        if self.sensor_measurements is None:
            raise ValueError("No sensor measurements found. Please add at least one dataset with sensor measurements before calling prepare().")
        
        sc = MinMaxScaler()
        sc.fit(self.sensor_measurements[self.train_indices])
        self.sensor_scaler = sc
        scaled_sensor_measurements = sc.transform(self.sensor_measurements)

        if masking_type is not None:
            if masking_type == 'grud':
                scaled_sensor_measurements = generate_GRUD_4D_input(scaled_sensor_measurements)
                lagged_sensor_measurements = generate_GRUD_4D_lags(scaled_sensor_measurements, self.lags)
            elif masking_type == 'double_channels':
                scaled_sensor_measurements = generate_channels(scaled_sensor_measurements)
                lagged_sensor_measurements = generate_lagged_sensor_measurements(scaled_sensor_measurements, self.lags)
            else:
                lagged_sensor_measurements = generate_lagged_sensor_measurements(scaled_sensor_measurements, self.lags)

        else:
            lagged_sensor_measurements = generate_lagged_sensor_measurements(scaled_sensor_measurements, self.lags)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # build datasets
        Xtrain = torch.tensor(lagged_sensor_measurements[self.train_indices], dtype=torch.float32, device=device)
        Ytrain = torch.tensor(scaled_data[self.train_indices], dtype=torch.float32, device=device)
        
        Xval = torch.tensor(lagged_sensor_measurements[self.val_indices], dtype=torch.float32, device=device)
        Yval = torch.tensor(scaled_data[self.val_indices], dtype=torch.float32, device=device)

        Xtest = torch.tensor(lagged_sensor_measurements[self.test_indices], dtype=torch.float32, device=device)
        Ytest = torch.tensor(scaled_data[self.test_indices], dtype=torch.float32, device=device)

        print("\nNumber of nan values in sensor measurements after masking:", np.isnan(scaled_sensor_measurements).sum())

        if verbose:
            print(f"\nData shapes after preparation:")
            print(f"Train:\t {Xtrain.shape} -> {Ytrain.shape}")
            print(f"Val:\t {Xval.shape} -> {Yval.shape}")
            print(f"Test:\t {Xtest.shape} -> {Ytest.shape}")   

        # return datasets
        if masking_type is not None and masking_type == 'mask_data':
            train_dataset = TimeSeriesDatasetWithMask(Xtrain, Ytrain)
            val_dataset = TimeSeriesDatasetWithMask(Xval, Yval)
            test_dataset = TimeSeriesDatasetWithMask(Xtest, Ytest)
        elif masking_type is not None and masking_type == 'mask_data_random':
            train_dataset = TimeSeriesDatasetWithMask(Xtrain, Ytrain, random=True)
            val_dataset = TimeSeriesDatasetWithMask(Xval, Yval, random=True)
            test_dataset = TimeSeriesDatasetWithMask(Xtest, Ytest, random=True)
        else:
            train_dataset = TimeSeriesDataset(Xtrain, Ytrain)
            val_dataset = TimeSeriesDataset(Xval, Yval)
            test_dataset = TimeSeriesDataset(Xtest, Ytest)

        Xtrain_mean = Xtrain.mean(dim=0, keepdim=True)

        if masking_type is not None and masking_type == 'grud':
            return train_dataset, val_dataset, test_dataset, Xtrain_mean

        return train_dataset, val_dataset, test_dataset

# data manager class
class ROMDataManager:

    def __init__(
        self,
        lags: int = 25,             # number of past time steps to use as input
        train_size: float = 0.8,    # percentage of data to use for training
        val_size: float = 0.1,      # percentage of data to use for validation
        test_size: float = 0.1,     # percentage of data to use for testing
    ):
        
        #check that the sizes sum to 1
        if not abs(train_size + val_size + test_size - 1.0) < 1e-8:
            raise ValueError("train_size, val_size and test_size must sum to 1")
        
        # store the parameters
        self.lags = lags
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # indices for train/val/test split
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # other params
        self._dataset_ids = []
        self.sensor_measurements = None
        self.data = None
        self._Vt_registry = {}
        self._preSVD_scaler_registry = {}   
        self._dataset_lengths = {}      
        self._dataset_spatial_shape = {}

    def add_data(
        self,
        data: DataInput,
        id: str,
        measurements: Optional[np.ndarray] = None,                      # sensor measurements (if None, will be loaded from data)
        compress: Union[None, bool, int] = True,                        # whether to apply SVD compression to the data (if True, compress to x components; if None or False, do not compress)
        verbose: bool = False
    ):
        #Data are of shape (ntrajectories, ntimes, nstates)
        
        # check that the id is unique
        if id in self._dataset_ids:
            raise ValueError(f"Dataset with id '{id}' already exists. Please choose a unique id for each dataset.")
        
        modes = parse_compress(compress) 
        data = get_data(data)
        dataset_spatial_shape = data.shape[-1]  # assume data is (ntraj, ntimes, spatial_dims...)

        # split in parameter dimension (ntrajectories) 
        if self.train_indices is None:
            self.train_indices = np.arange(0, int(data.shape[0] * self.train_size))
            self.val_indices = np.arange(int(data.shape[0] * self.train_size), int(data.shape[0] * (self.train_size + self.val_size)))
            self.test_indices = np.arange(int(data.shape[0] * (self.train_size + self.val_size)), data.shape[0])

        if measurements is not None:
            measurements = measurements.reshape(measurements.shape[0], measurements.shape[1], -1)  # ensure measurements are (ntrajectories, ntimes, num_sensors)

        # dimensionality reduction
        if modes > 0:
            # Always treat data as 3D: (ntraj, ntimes, nstates)
            ntraj, ntimes, nstates = data.shape
            data_reshaped = data.reshape(-1, nstates)  # (ntraj*ntimes, nstates)
            train_rows = np.concatenate([np.arange(i * ntimes, (i + 1) * ntimes) for i in self.train_indices])
            sc = StandardScaler()
            sc.fit(data_reshaped[train_rows])
            data_scaled = sc.transform(data_reshaped)
            U, S, Vt = randomized_svd(data_scaled[train_rows], n_components=modes, n_iter='auto')

            #plot singular values decay
            reconstruction = U @ np.diag(S) @ Vt
            # Use train_rows as in SVD step
            residual_energy = np.linalg.norm(data_scaled[train_rows] - reconstruction) ** 2 \
                        / np.linalg.norm(data_scaled[train_rows]) ** 2
            
            data_scaled = data_scaled @ Vt.T
            data = data_scaled.reshape(data.shape[0], data.shape[1], -1)

            if verbose:
                plt.semilogy(S[:modes], color = 'magenta', linewidth = 2)
                plt.title("Singular Values Decay \n")
                plt.show()

            print(f"Residual energy of {id} after keeping {modes} modes: {residual_energy*100:.4f} %")


        # commit to self
        self._dataset_ids.append(id)
        self._dataset_spatial_shape[self._dataset_ids[-1]] = dataset_spatial_shape
        if measurements is not None:
            if self.sensor_measurements is None:
                self.sensor_measurements = measurements
            else:
                self.sensor_measurements = np.hstack((self.sensor_measurements, measurements))
        if modes > 0:
            self._preSVD_scaler_registry[self._dataset_ids[-1]] = sc
            self._Vt_registry[self._dataset_ids[-1]] = Vt
        self._dataset_lengths[self._dataset_ids[-1]] = data.shape[-1]  # save spatial dim to registry
        if self.data is None:
            self.data = data
        else:
            self.data = np.concatenate((self.data, data), axis=-1)


    def prepare(self, verbose = False, masking_type = None, unused_data = None):
        # This method will prepare the data for training by applying scaling, generating lagged features, and creating PyTorch datasets for train/val/test splits. It will also handle masking of sensor measurements if specified.

        if self.data is None:
            raise ValueError("No data found. Please add at least one dataset before calling prepare().")

        print(f"\nDatasets: {self._dataset_ids}")
        if unused_data is not None:
            excluded_ids = {unused_data} if isinstance(unused_data, str) else set(unused_data)
            print(f"Dataset NOT used: {sorted(excluded_ids)}")
            self._active_dataset_ids = [ids for ids in self._dataset_ids if ids not in excluded_ids]
        else:
            self._active_dataset_ids = list(self._dataset_ids)

        if not self._active_dataset_ids:
            raise ValueError("All datasets were excluded via unused_data; no output targets remain.")
    
        data = self.data.reshape(-1,self.data.shape[-1])  # reshape to (ntraj*ntimes, nstates) for scaling
        sc = MinMaxScaler()
        sc.fit(data[self.train_indices].reshape(-1, data.shape[-1]))
        self.data_scaler = sc
        scaled_data = sc.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

        if self.sensor_measurements is None:
            raise ValueError("No sensor measurements found. Please add at least one dataset with sensor measurements before calling prepare().")
        
        sc = MinMaxScaler()
        sc.fit(self.sensor_measurements[self.train_indices].reshape(-1, self.sensor_measurements.shape[-1]))
        self.sensor_scaler = sc
        scaled_sensor_measurements = sc.transform(self.sensor_measurements.reshape(-1, self.sensor_measurements.shape[-1])).reshape(self.sensor_measurements.shape)

        if masking_type is not None:
            if masking_type == 'grud':
                scaled_sensor_measurements = generate_GRUD_4D_input(scaled_sensor_measurements)
                lagged_sensor_measurements = generate_GRUD_4D_lags(scaled_sensor_measurements, self.lags)
            elif masking_type == 'double_channels':
                scaled_sensor_measurements = generate_channels(scaled_sensor_measurements)
                lagged_sensor_measurements = generate_lagged_sensor_measurements(scaled_sensor_measurements, self.lags)
            else:
                lagged_sensor_measurements = generate_lagged_sensor_measurements(scaled_sensor_measurements, self.lags) 
        else:
            lagged_sensor_measurements = generate_lagged_sensor_measurements(scaled_sensor_measurements, self.lags)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Robust train/val/test split
        n_total = lagged_sensor_measurements.shape[0]
        n_train = int(n_total * self.train_size)
        n_val = int(n_total * self.val_size)
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_train + n_val)
        test_indices = np.arange(n_train + n_val, n_total)

        Xtrain = torch.tensor(lagged_sensor_measurements[train_indices], dtype=torch.float32, device=device)
        Ytrain = torch.tensor(scaled_data.reshape(-1, scaled_data.shape[-1])[train_indices], dtype=torch.float32, device=device)
        Xval = torch.tensor(lagged_sensor_measurements[val_indices], dtype=torch.float32, device=device)
        Yval = torch.tensor(scaled_data.reshape(-1, scaled_data.shape[-1])[val_indices], dtype=torch.float32, device=device)
        Xtest = torch.tensor(lagged_sensor_measurements[test_indices], dtype=torch.float32, device=device)
        Ytest = torch.tensor(scaled_data.reshape(-1, scaled_data.shape[-1])[test_indices], dtype=torch.float32, device=device)

        print("\nNumber of nan values in sensor measurements after masking:", np.isnan(scaled_sensor_measurements).sum())

        if verbose:
            print(f"\nData shapes after preparation:")
            print(f"Train:\t {Xtrain.shape} -> {Ytrain.shape}")
            print(f"Val:\t {Xval.shape} -> {Yval.shape}")
            print(f"Test:\t {Xtest.shape} -> {Ytest.shape}")

        # return datasets
        if masking_type is not None and masking_type == 'mask_data':
            train_dataset = TimeSeriesDatasetWithMask(Xtrain, Ytrain)
            val_dataset = TimeSeriesDatasetWithMask(Xval, Yval)
            test_dataset = TimeSeriesDatasetWithMask(Xtest, Ytest)
        elif masking_type is not None and masking_type == 'mask_data_random':
            train_dataset = TimeSeriesDatasetWithMask(Xtrain, Ytrain, random=True)
            val_dataset = TimeSeriesDatasetWithMask(Xval, Yval, random=True)
            test_dataset = TimeSeriesDatasetWithMask(Xtest, Ytest, random=True)
        else:
            train_dataset = TimeSeriesDataset(Xtrain, Ytrain)
            val_dataset = TimeSeriesDataset(Xval, Yval)
            test_dataset = TimeSeriesDataset(Xtest, Ytest)

        Xtrain_mean = Xtrain.mean(dim=0, keepdim=True)
        if masking_type is not None and masking_type == 'grud':
            return train_dataset, val_dataset, test_dataset, Xtrain_mean
        return train_dataset, val_dataset, test_dataset
                                                                                             

# TimeseriesDataset
class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        # input X should be of shape (N, lags, num_sensors) and Y should be of shape (N, state dims)
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        # return the sensor sequence and corresponding full-state measurement for the given index
        return self.X[index], self.Y[index]
    
    def __len__(self):        
        # return the number of samples in the dataset
        return self.len

class TimeSeriesDatasetWithMask(torch.utils.data.Dataset):

    def __init__(self, X, Y, random=False):
        # X should be of shape (N, lags, num_sensors) nan replaced with 0
        # M should be of shape (N, lags, num_sensors) and indicate which sensor measurements are observed (1) vs masked (0)
        # Y should be of shape (N, state dims)

        self.X = X
        self.M = ~torch.isnan(X)            # mask indicating observed (1) vs masked
        if not random:
            # replace nan with 0
            self.X[torch.isnan(self.X)] = 0     # replace nan with 0
        else:
            # replace nan with random values between -0.5 and 1.5 
            # (since we will apply MinMaxScaler to the sensor measurements, 
            # which scales them to be between 0 and 1, we can use a wider range 
            # for the random values to ensure they are outside the normal range of the data)  
            self.X[torch.isnan(self.X)] = torch.rand(torch.isnan(self.X).sum(), device=X.device) * 2 - 0.5
        self.Y = Y
        self.len = X.shape[0]

        self.M = self.M.float()  # convert mask to float for use in models that expect float inputs
        self.X = self.X.float()  # ensure X is float for use in models
        self.Y = self.Y.float()  # ensure Y is float for use in models

    def __getitem__(self, index):
        # return the sensor sequence, mask, and corresponding full-state measurement for the given index
        return self.X[index], self.M[index], self.Y[index]
    
    def __len__(self):        
        # return the number of samples in the dataset
        return self.len     
        
# utility function

def parse_compress(compress):
    # normalize the compress parameter to an integer number of components to keep
    if compress is None or compress is False:
        return 0
    elif compress is True:
        return DEFAULT_MODES
    elif isinstance(compress, int) and compress >= 0:
        return compress
    raise ValueError("compress must be None, False, True, or a non-negative integer")

def get_data_npz(file_path):
    # load data from a .npz file
    data = np.load(file_path)
    if len(data.files) == 0:
        raise ValueError(f"No arrays found in {file_path}. Please ensure the .npz file contains at least one array.")
    if len(data.files) > 1:
        print(f"Warning: Multiple arrays found in {file_path}. Only the first array will be used.")
    return data[data.files[0]]

def get_data_npy(file_path):
    # load data from a .npy file
    return np.load(file_path)
    
def get_data(data: DataInput):
    # get the data as a numpy array, whether it's provided as a file path, numpy array, or torch tensor
    #file
    if isinstance(data, str):
        if data.endswith('.npy'):
            return get_data_npy(data)
        elif data.endswith('.npz'):
            return get_data_npz(data)
        else:
            raise ValueError(f"Unsupported file format for {data}. Only .npy and .npz files are supported.")
    #data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    # if it's not a supported type, raise an error
    raise ValueError("data must be a file path (str), numpy array, or torch tensor")

def generate_random_sensor_locations(data, num_sensors, seed):
    # generate random sensor locations given the spatial shape of the data
    spatial_shape = data.shape[1:]  # assume data is (time, spatial_dims...)
    spatial_points = np.prod(spatial_shape)
    if seed is not None:
        np.random.seed(seed)
    random_indices = np.random.choice(spatial_points, size=num_sensors, replace=False)
    sensor_locations = []
    for idx in random_indices:
        sensor_location = []
        for dim in reversed(spatial_shape):
            sensor_location.append(idx % dim)
            idx //= dim
        sensor_locations.append(tuple(reversed(sensor_location)))
    return sensor_locations

def get_sensor_measurements(
    data, 
    id, 
    sensor_number, 
    random, 
    stationary, 
    mobile, 
    measurements, 
    seed,
    train_indices = None,
    imputer: str = None
    ):
    
    sensor_summary = []
    sensor_measurements = []

    # randomly selected stationary sensors
    if random is not None:
        if isinstance(random,int):
            random_sensor_locations = generate_random_sensor_locations(data, random, seed)
            for sensor_coord in random_sensor_locations:
                sensor_summary.append([id, next(sensor_number), 'stationary - random', sensor_coord])
                sensor_measurements.append(data[(slice(None),) + sensor_coord]) # all timesteps at that (i,j) location
        else:
            raise ValueError("random must be an integer specifying the number of random sensors to select")
        
    # selected stationary sensors
    if stationary is not None:
        if isinstance(stationary, tuple):
            stationary = [stationary]
        if all(isinstance(s, tuple) for s in stationary):
            for sensor_coord in stationary:
                sensor_summary.append([id, next(sensor_number), 'stationary', sensor_coord])
                sensor_measurements.append(data[(slice(None),) + sensor_coord]) # all timesteps at that (i,j) location
        else:
            raise ValueError("stationary must be a tuple or list of tuples specifying the coordinates of the stationary sensors")
        
    # selected mobile sensors
    if mobile is not None:
        if isinstance(mobile[0], tuple):
            mobile = [mobile]
        if all(isinstance(m, list) for m in mobile):
            for mobile_sensor_coords in mobile:
                if len(mobile_sensor_coords) != data.shape[0]:
                    raise ValueError("Each mobile sensor must have coordinates specified for each time step")
                sensor_summary.append([id, next(sensor_number), 'mobile', mobile_sensor_coords])
                sensor_measurements.append([
                    data[timestep, sensor_coord] for timestep, sensor_coord in enumerate(mobile_sensor_coords)
                ])
        else:
            raise ValueError("mobile must be a list of lists of tuples specifying the coordinates of the mobile sensors for each time step")
        
    # sensor data
    if measurements is not None:
        if isinstance(measurements, torch.Tensor):
            measurements = measurements.detach().cpu().numpy()
        if not isinstance(measurements, np.ndarray):
            raise ValueError("measurements must be a numpy array or torch tensor of shape (time, num_sensors)")
        if measurements.shape[0] != data.shape[0]:
            raise ValueError("measurements must have the same number of time steps as the data")

        if imputer is not None:
            if train_indices is None:
                raise ValueError("train_indices must be provided for median imputation")
            if imputer == 'median':
                measurements = median_imputer_single_dataset(measurements, train_indices)
            elif imputer == 'mean':
                measurements = mean_imputer_single_dataset(measurements, train_indices)
            elif imputer == 'last_value':
                measurements = last_value_imputer_single_dataset(measurements, train_indices)
            else:
                raise ValueError("Unsupported imputation strategy. Supported strategies are 'median', 'mean', and 'last_value'.")      

        print("Nuber of nan values in measurements after imputation:", np.isnan(measurements).sum(), "\n")      
        
        measurements = measurements.T  # transpose to get shape (num_sensors, time)

        for row in measurements:
            sensor_summary.append([id, next(sensor_number), 'provided', None])
            sensor_measurements.append(row)

    # transpose sensor_measurements to have time on axis 0 and num_sensors on axis 1
    if len(sensor_summary) == 0:
        sensor_summary = pd.DataFrame(columns=['data id', 'sensor number', 'type', 'loc/traj'])
    else:
        sensor_summary = pd.DataFrame(sensor_summary, columns=['data id', 'sensor number', 'type', 'loc/traj'])

    if len(sensor_measurements) == 0:
        sensor_measurements = None
        sensor_measurements_df = None
    else:
        sensor_measurements = np.array(sensor_measurements).T
        labels = (
            sensor_summary['data id'].astype(str).str.cat(sensor_summary['sensor number'].astype(str), sep='-')
        )
        labels.name = None
        sensor_measurements_df = pd.DataFrame(sensor_measurements, columns=labels)

    # return a dict with the sensor summary and measurements
    return {
        'sensor_summary': sensor_summary,
        'sensor_measurements': sensor_measurements, 
        'sensor_measurements_df': sensor_measurements_df
    }

def generate_lagged_sensor_measurements(sensor_measurements, lags):
    # generate lagged sensor measurements for use as model inputs
    # Accepts 2D (ntimes, ns) or 3D (ntraj, ntimes, ns) input

    if sensor_measurements.ndim == 2:
        num_timesteps, num_sensors = sensor_measurements.shape
        # concatenate zeros to the beginning
        padded = np.concatenate((np.zeros((lags, num_sensors)), sensor_measurements), axis=0)
        lagged_sequences = np.empty((num_timesteps, lags, num_sensors), dtype=sensor_measurements.dtype)
        for i in range(num_timesteps):
            lagged_sequences[i] = padded[i:i+lags, :]
        return lagged_sequences
    
    elif sensor_measurements.ndim == 3:
        ntraj, ntimes, ns = sensor_measurements.shape
        xshape = (ntraj, ntimes, lags, ns)
        lagged_sequences = np.empty(xshape, dtype=sensor_measurements.dtype)
        for traj in range(ntraj):
            padded = np.concatenate((np.zeros((lags, ns)), sensor_measurements[traj]), axis=0)
            for i in range(ntimes):
                lagged_sequences[traj, i] = padded[i:i+lags, :]
        return lagged_sequences.reshape(ntraj*ntimes, lags, ns)  # reshape to  (trajectories*timesteps, lags, num_sensors) for use in training
    
    else:
        raise ValueError("sensor_measurements must be 2D or 3D array")

def generate_GRUD_4D_lags(sensor_measurements, lags):
    # generate lagged features for GRUD, which expects a 4D input of shape (N, channels, lags, num_sensors)
    
    channels, num_timesteps, num_sensors = sensor_measurements.shape

    #concatenate zeros to the beginning of the sensor measurements along the temporal axis to create lagged features
    sensor_measurements = np.concatenate((np.zeros((channels, lags, num_sensors)), sensor_measurements), axis=1)
    lagged_sequences = np.empty((num_timesteps, channels, lags, num_sensors))
    for i in range(num_timesteps):
        # sensor_measurements[:,i:i+lags,:] has shape (channels, lags, num_sensors)
        lagged_sequences[i,:,:,:] = sensor_measurements[:,i:i+lags,:]
    return lagged_sequences
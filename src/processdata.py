import torch                                    # type: ignore
import numpy as np                              # type: ignore
import matplotlib.pyplot as plt                 # type: ignore
import imageio                                  # type: ignore
from IPython.display import clear_output as clc # type: ignore
from IPython.display import display             # type: ignore

from .utils import mae, mse, mre, num2p

class TimeSeriesDataset(torch.utils.data.Dataset):

    # Initialize the dataset with input data X and target data Y
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    # Return the length of the dataset
    def __len__(self):
        return self.len

    # Return the input and target data for a given index
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class MaskedTimeSeriesDataset(torch.utils.data.Dataset):

    # Initialize the dataset with input data X and target data Y, and a mask indicating which time steps are valid
    def __init__(self, X, mask, Y):
        self.X    = X
        self.mask = mask
        self.Y    = Y
        self.len  = X.shape[0]

    # Return the length of the dataset
    def __len__(self):
        return self.len

    # Return the input and target data for a given index, along with the corresponding mask
    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.Y[idx]
    

#Padding Function
def PaddingGRUD(data, lag):

    # Extract time-series of lenght equal to lag from longer time series in data,
    # whose dimension is (number of trajectories, number of channels, sequence length, data shape)
    ntrajectories, nchannels, ntimes, nsensors = data.shape
    data_padded = torch.zeros(ntrajectories * ntimes, nchannels, lag, nsensors)

    # Loop through each trajectory and time step in the input data, 
    # and pad with zeros if the current time step is less than the lag, 
    # or take the last 'lag' time steps if the current time step is greater than or equal to the lag
    for i in range(ntrajectories):
        for j in range(1, ntimes + 1):
            if j < lag:                                         # pad the beginning with zeros
                data_padded[i * ntimes + j - 1, :, -j:, :] = data[i, :, :j, :]
            else:                                               # full window
                data_padded[i * ntimes + j - 1] = data[i, :, j - lag:j, :]

    return data_padded


#Padding Function
def Padding(data,lag):

    # Extract time-series of lenght equal to lag from longer time series in data,
    # whose dimension is (number of time series, sequence length, data shape)

    data_padded = torch.zeros(data.shape[0]*data.shape[1], lag, data.shape[2]) # Initialize a new tensor to hold the padded data, with dimensions (number of time series * sequence length, lag, data shape)
    
    for i in range(data.shape[0]):                                          # Loop through each time series in the input data
        for j in range(1, data.shape[1]+1):                                 # Loop through each time step in the current time series
            if j < lag:                                                     # If the current time step is less than the lag, pad with zeros
                data_padded[i*data.shape[1]+j-1, -j:, :] = data[i, :j, :]
            else:                                                           # If the current time step is greater than or equal to the lag, take the last 'lag' time steps
                data_padded[i*data.shape[1]+j-1] = data[i, j-lag:j, :]

    return data_padded

# Padding Function for boolean key masks
def PaddingMask(mask, lag):

    # Extract time-series of lenght equal to lag from longer time series in data,
    # whose dimension is (number of time series, sequence length, data shape)

    ntrajectories, ntimes = mask.shape
    out = torch.zeros(ntrajectories * ntimes, lag, dtype=torch.bool)        # Initialize a new tensor to hold the padded masks, with dimensions (number of time series * sequence length, lag)

    for i in range(ntrajectories):
        for j in range(1, ntimes + 1):
            if j < lag:                                         # pad the beginning with False
                out[i * ntimes + j - 1, -j:] = mask[i, :j]      # leading positions stay False
            else:                                               # full window
                out[i * ntimes + j - 1] = mask[i, j - lag:j]

    return out

def trajectory(
    yt,         # trajectory with dimension (sequence length, data shape)
    plot,       # related plot function for a snapshot, which takes as input a snapshot with dimension (data shape) and produces a plot
    title = None, fontsize = None, figsize = None, axis = False, save = False, name = 'gif'
    ):

    arrays = []
    
    # Loop through each time step in the trajectory,
    for i in range(yt.shape[0]):
        plt.figure(figsize = figsize)           # Create a new figure for each time step, with the specified size
        plot(yt[i])                             # Call the provided plot function to visualize the current time step, which takes as input a snapshot with dimension (data shape) and produces a plot
        plt.title(title, fontsize = fontsize)   
        if not axis:
            plt.axis('off')
        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    # If the save flag is set to True, save the generated plots as a GIF file with the specified name, using the imageio library to create the GIF from the list of arrays representing each frame of the animation
    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)
        
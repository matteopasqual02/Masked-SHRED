import torch                    #type: ignore
import torch.nn as nn           #type: ignore
import numpy as np               #type: ignore

# MLP
class MLP(nn.Module):

    def __init__(
        self,
        in_dim: int,        # Input dimension of the MLP
        out_dim: int,       # Output dimension of the MLP
        n_layers: int,      # Number of layers in the MLP
        dropout: float,     # Dropout rate to apply after each layer
        device: str = 'cpu' # Device to run the MLP on (e.g., 'cpu' or 'cuda')
    ):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Create a list of linear layers for the MLP
        sizes = list()
        sizes.extend(np.logspace(
            np.log2(in_dim), np.log2(out_dim), num=n_layers+1, base=2, dtype=int
        ).tolist())
        sizes[0] = in_dim  # Ensure the first layer has the correct input dimension
        sizes[-1] = out_dim # Ensure the last layer has the correct output dimension

        #Define the linear layers of the MLP
        self.layers = []
        #Iterate through the sizes and create linear layers with ReLU activations in between
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx+1]))
            if idx != len(sizes)-2:  
                self.layers.append(nn.ReLU())
        
        model = nn.Sequential(*self.layers)     # Create a sequential model from the list of layers
        self.model = model.to(device)           # Move the model to the specified device

    def forward(self, x):
        sindy_loss = x.get("sindy_loss", None)  # Extract the 'sindy_loss' from the input dictionary, if it exists
        x = x["output"]
        out = self.model(x)         # Pass the input through the MLP
        out = self.dropout(out)     # Apply dropout to the output of the MLP

        # return the output of the MLP along with the 'sindy_loss' if it exists
        return {
            "output": out,
            "sindy_loss": sindy_loss
        }
    
# 1D CNN
class CNN(nn.Module):

    def __init__(
        self,
        in_dim: int,        # Input dimension of the CNN
        out_dim: int,       # Output dimension of the CNN
        n_layers: int,      # Number of layers in the CNN
        dropout: float,     # Dropout rate to apply after each layer
        device: str = 'cpu' # Device to run the CNN on (e.g., 'cpu' or 'cuda')
    ):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Create a list of linear layers for the MLP
        sizes = list()
        sizes.extend(np.logspace(
            np.log2(in_dim), np.log2(out_dim), num=n_layers+1, base=2, dtype=int
        ).tolist())
        sizes[0] = in_dim  # Ensure the first layer has the correct input dimension
        sizes[-1] = out_dim # Ensure the last layer has the correct output dimension

        #Define the linear layers of the MLP
        self.layers = []
        #Iterate through the sizes and create linear layers with ReLU activations in between
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Conv1d(sizes[idx], sizes[idx+1], kernel_size=3, padding=1))
            if idx != len(sizes)-2:  
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)     # Create a sequential model from the list of layers
        self.model = model.to(device)           # Move the model to the specified device


    def forward(self, x):
        sindy_loss = x.get("sindy_loss", None)
        x = x["output"]

        batch_size, forecast_length, sequence_length, hidden_dim = x.shape

        # (b, f, s, d) -> (b, d, f, s)
        x = x.permute(0, 3, 1, 2)
        # (b, d, f, s) -> (b, d, f*s)
        x = x.reshape(batch_size, hidden_dim, forecast_length * sequence_length)

        out = self.model(x)
        out = self.dropout(out)

        # out shape assumed: (b, out_dim, f*s)
        b, out_dim, _ = out.shape
        # (b, o, f*s) -> (b, o, f, s)
        out = out.reshape(b, out_dim, forecast_length, sequence_length)
        # (b, o, f, s) -> (b, f, s, o)
        out = out.permute(0, 2, 3, 1)

        return {
            "output": out, 
            "sindy_loss": sindy_loss
        }
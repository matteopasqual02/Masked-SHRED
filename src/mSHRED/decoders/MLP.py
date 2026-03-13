import torch                    #type: ignore
import torch.nn as nn           #type: ignore
import torch.nn.functional as F #type: ignore
import numpy as np               #type: ignore

# MLP
class MLP(nn.Module):

    def __init__(
        self,
        input_size: int,            #input size
        output_size: int,           #output size
        hidden_sizes: np.ndarray,   #hidden layer sizes
        dropout: float = 0.1,       #dropout rate
    ):
        
        super().__init__()

        # Store parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        sizes = [input_size] + list(hidden_sizes) + [output_size]   #layer sizes

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)   #create linear layers
        ])

    def forward(self, x):

        # Pass through each layer
        for i,layer in enumerate(self.layers):
            x = layer(x)                    #apply linear layer
            if i < len(self.layers) - 1:    #apply activation and dropout for hidden layers
                x = F.gelu(x)
                x = self.dropout(x)

        return x
    
    @property
    def model_name(self):
        return "MLP"
    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
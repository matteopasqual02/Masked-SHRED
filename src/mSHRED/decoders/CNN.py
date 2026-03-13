import torch                    #type: ignore
import torch.nn as nn           #type: ignore
import torch.nn.functional as F #type: ignore
import numpy as np               #type: ignore

# 1D CNN decoder
class CNN(nn.Module):

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

        channels = [input_size] + list(hidden_sizes) + [output_size]

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

        # Kernel size 1 preserves sequence length while transforming channels.
        self.layers = nn.ModuleList([
            nn.Conv1d(channels[i], channels[i + 1], kernel_size=1)
            for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        # Accept (batch, channels) and project as a length-1 sequence.
        x = x.unsqueeze(-1)

        # Pass through each layer
        for i,layer in enumerate(self.layers):
            x = layer(x)                    #apply linear layer
            if i < len(self.layers) - 1:    #apply activation and dropout for hidden layers
                x = F.gelu(x)
                x = self.dropout(x)

        return x.squeeze(-1)
    
    @property
    def model_name(self):
        return "CNN"
    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



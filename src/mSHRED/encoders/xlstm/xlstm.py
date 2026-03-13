import torch                    #type: ignore
import torch.nn as nn           #type: ignore
import torch.nn.functional as F #type: ignore

#import the sLSTM cell
from .slstm import sLSTM
from .mlstm import mLSTM

#xlstm module
class XLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        type: str = 'sLSTM',
    ):
        
        super().__init__()

        #create the xlstm 
        if type == 'sLSTM':
            self.model = sLSTM(input_size, hidden_size, num_layers)
        elif type == 'mLSTM':
            self.model = mLSTM(input_size, hidden_size, num_layers)
        else:
            raise ValueError(f"Unsupported type: {type}")
        
    def forward(self, x, **kwargs):

        #pass the input through the xlstm model and return the output
        output = self.model(x)  # (batch_size, seq_len, hidden_size)

        # Return only the last timestep's output for each sequence in the batch
        # SHAPE: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        output = output[:, -1, :]  # Extract the output corresponding to the last timestep for each sequence

        assert output.dim() == 2, "Expected output shape (batch_size, hidden_size), got {}".format(output.shape)

        # Return the output of the transformer encoder
        return output
        
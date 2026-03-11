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

        # Add an extra dimension for compatibility with the expected output shape
        output = output.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_size)

        return {
            "sequence_output": output,                  # FULL OUTPUT   (batch_size, 1, seq_len, hidden_size)
            "final_hidden_state": output[:, :, -1, :],  # LAST TIMESTEP (batch_size, 1, hidden_size)
            "output": output,
            "sindy_loss": None                          # Placeholder for SINDy loss
        }
        
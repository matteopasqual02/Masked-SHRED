import torch                            # type: ignore
import torch.nn as nn                   # type: ignore

class LSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = hidden_layers,
            batch_first = True)
        

    def forward(self, x):
        # Init h0 and c0 on the same device/dtype as input.
        h0 = x.new_zeros((self.hidden_layers, x.size(0), self.hidden_size))
        c0 = x.new_zeros((self.hidden_layers, x.size(0), self.hidden_size))


        # Encoder forward pass: return all hidden states 
        # out shape: (batch_size, seq_len, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)

        out, (hn, cn) = self.encoder(x, (h0, c0))
        out = out.permute(1,0,2)  # (seq_len, batch_size, hidden_size)

        # print("LSTM output shape:", out.shape)  # Debug print
        # print("LSTM hn shape:", hn.shape)       # Debug print

        #check that the last out is the same as the last hn
        assert torch.allclose(out[-1], hn[-1]), f"Last output does not match last hidden state: {out[-1]} vs {hn[-1]}"
        assert out.dim() == 3, f"LSTM output should be 3D (seq_len, batch_size, hidden_size), but got {out.dim()}D"

        return out

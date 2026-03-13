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

        # Encoder forward pass: return last layer hidden state as latent embedding.
        _, (hn, _) = self.encoder(x, (h0, c0))
        out = hn[-1].view(-1, self.hidden_size)

        assert out.dim() == 2, "Expected output shape (batch_size, hidden_size), got {}".format(out.shape)
        return out

import torch            #type: ignore
import torch.nn as nn   #type: ignore
import numpy as np      #type: ignore
from types import SimpleNamespace

from .transformer import Transformer
from .sdn import MLP, CNN

class TSHRED(nn.Module):

    def __init__(self,args):

        super().__init__()

        if isinstance(args, dict):
            args = SimpleNamespace(**args)

        # Encoder choice 
        if args.encoder == "vanilla_transformer":
            self.encoder = Transformer(
                d_model=args.d_model,
                n_heads=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                input_length=args.input_length,
                num_encoder_layers=args.encoder_depth,
                norm_first=False,
                layer_norm_eps=1e-5,
                bias=True,
                device=args.device,
            )
        else:
            raise NotImplementedError(f"Encoder {args.encoder} not implemented")

        # SDN choice
        if args.decoder == "cnn":
            self.decoder = CNN(
                in_dim=args.hidden_size,
                out_dim=args.output_size,
                n_layers=args.decoder_depth,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.decoder == "mlp":
            self.decoder = MLP(
                in_dim=args.hidden_size,
                out_dim=args.output_size,
                n_layers=args.decoder_depth,
                dropout=args.dropout,
                device=args.device,
            )
        else:
            raise NotImplementedError(f"Decoder {args.decoder} not implemented")
        
    def forward(self, src):

        # Pass the input through the encoder
        encoded = self.encoder(src)
        # Pass the encoded output through the decoder
        decoded = self.decoder(encoded) 

        return decoded

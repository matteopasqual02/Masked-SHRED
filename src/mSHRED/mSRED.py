import torch.nn as nn                       #type: ignore
from types import SimpleNamespace           #type: ignore


class mSHRED(nn.Module):

    def __init__(self,args):

        super().__init__()
        if isinstance(args, dict):
            args = SimpleNamespace(**args)

        self.encoder = None
        self.decoder = None

        # Encoder choice
        if args.encoder == "transformer":
            from .encoders.transformer import Transformer
            self.encoder = Transformer(
                d_model=args.input_size,
                hidden_size=args.hidden_size,
                n_heads=args.n_heads,
                num_encoder_layers=args.encoder_depth,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                layer_norm_eps=1e-5,
                norm_first=False,
                bias=True,
                input_length=args.input_length,
            )
        elif args.encoder == "transformer_hidden_mask":
            from .encoders.transformer_hidden_mask import TransformerHiddenMask
            self.encoder = TransformerHiddenMask(
                d_model=args.input_size,
                hidden_size=args.hidden_size,
                n_heads=args.n_heads,
                num_encoder_layers=args.encoder_depth,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                layer_norm_eps=1e-5,
                norm_first=False,
                bias=True,
                input_length=args.input_length,
            )
        elif args.encoder == "transformer_attn_mask":
            from .encoders.transformer_attn_mask import TransformerAttnMask
            self.encoder = TransformerAttnMask(
                d_model=args.input_size,
                hidden_size=args.hidden_size,
                n_heads=args.n_heads,
                num_encoder_layers=args.encoder_depth,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                layer_norm_eps=1e-5,
                norm_first=False,
                bias=True,
                input_length=args.input_length,
            )
        elif args.encoder == "xlstm":
            from .encoders.xlstm.xlstm import XLSTM
            self.encoder = XLSTM(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                type=args.xlstm_type,
            )     
        elif args.encoder == "lstm":
            from .encoders.LSTM import LSTM
            self.encoder = LSTM(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                hidden_layers=args.num_layers,
            )    
        elif args.encoder == "grud":
            from .encoders.GRUD import GRUD
            self.encoder = GRUD(
                input_size=args.input_size,
                cell_size=args.input_size,
                hidden_size=args.hidden_size,
                X_mean=args.X_mean,
                output_last=True,
            )
        elif args.encoder == "att_grud":   
            from .encoders.attnGRUD import AttnGRUD
            self.encoder = AttnGRUD(
                input_size=args.input_size,
                cell_size=args.input_size,
                hidden_size=args.hidden_size,
                X_mean=args.X_mean,
                output_last=True,
            )
        else:
            raise NotImplementedError(f"Encoder {args.encoder} not implemented")
        
        # Decoder choice
        if args.decoder == "mlp":
            from .decoders.MLP import MLP
            self.decoder = MLP(
                input_size=args.hidden_size,
                output_size=args.output_size,
                hidden_sizes=args.decoder_hidden_sizes,
                dropout=args.dropout,
            )
        elif args.decoder == "cnn":
            from .decoders.CNN import CNN
            self.decoder = CNN(
                input_size=args.hidden_size,
                output_size=args.output_size,
                hidden_sizes=args.decoder_hidden_sizes,
                dropout=args.dropout,
            )
        else:
            raise NotImplementedError(f"Decoder {args.decoder} not implemented")
        

    def forward(self, x, mask=None):

        # Check if encoder and decoder are implemented
        if self.encoder is None:
            raise NotImplementedError("Encoder not implemented")
        if self.decoder is None:
            raise NotImplementedError("Decoder not implemented")
        
        if mask is not None:
            encoded = self.encoder(x, mask)
        else:
            encoded = self.encoder(x)

        return self.decoder(encoded)
        
    def freeze(self):

        # Freeze the model parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):

        # Unfreeze the model parameters
        self.train()
        for param in self.parameters():
            param.requires_grad = True

        
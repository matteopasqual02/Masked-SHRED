import torch                                #type: ignore
import torch.nn as nn                       #type: ignore
from torch.utils.data import DataLoader     #type: ignore
import numpy as np                          #type: ignore
from types import SimpleNamespace           #type: ignore
from tqdm import tqdm                       #type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau      #type: ignore

from .transformer import Transformer
from .xlstm.xlstm import XLSTM
from .sdn import MLP, CNN
from ..utils import num2p, mre, mse

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
        elif args.encoder == "xlstm":
            self.encoder = XLSTM(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                type=args.xlstm_type,
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
        
    def forward(self, src, mask=None):
        
        # Pass the input through the encoder, optionally with key-padding mask
        encoded = self.encoder(src, src_mask=mask)
        # Pass the encoded output through the decoder
        decoded = self.decoder(encoded)

        # Unwrap dict output -> (batch, out_dim)
        return decoded["output"]
    
    def freeze(self):
        self.eval()  # Set the model to evaluation mode
        for param in self.parameters():
            param.requires_grad = False  # Freeze the parameters by setting requires_grad to False

    def unfreeze(self):
        self.train()  # Set the model to training mode
        for param in self.parameters():
            param.requires_grad = True  # Unfreeze the parameters by setting requires_grad to True

def fit(
    model: TSHRED,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    epochs: int = 4000,
    optimizer: torch.optim.Optimizer = torch.optim.Adam,
    lr: float = 1e-3,
    loss_fn = mse,
    loss_output = mre,
    formatter = num2p,
    verbose: bool = False,
    patience: int = 100,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 20,
    min_lr: float = 1e-6,
):
    #init the data, the optimizer, params ...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )
    train_errors = []
    val_errors = []
    patience_counter = 0
    best_params = model.state_dict()

    #train loop 
    pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        
        #for each epoch
        for k, data in enumerate(train_loader):     #loop over the batches
            model.train()                           #set the model to training mode

            #define the closure function for the optimizer step
            def closure():
                optimizer.zero_grad()                 #zero the gradients

                # Support both (X, Y) and (X, mask, Y) datasets
                # if dim == 3, we assume the dataset returns (X, mask, Y) and we pass the mask to the model for a masked forward pass
                if len(data) == 3:
                    output = model(data[0], mask=data[1])   #masked forward pass
                    loss = loss_fn(data[2], output)
                # else, we assume the dataset returns (X, Y) and we perform a standard forward pass without masking
                else:
                    output = model(data[0])               #forward pass
                    loss = loss_fn(data[1], output)  #compute the loss

                loss.backward()                       #backward pass
                return loss
            optimizer.step(closure)                 #optimizer step

        #compute the training and validation errors
        model.eval()                            #set the model to evaluation mode
        with torch.no_grad():                   #disable gradient computation

            #train error
            ytrue_train = train_dataset.Y
            train_mask = getattr(train_dataset, 'mask', None)
            ypred_train = model(train_dataset.X, mask=train_mask)
            train_error = loss_output(ytrue_train, ypred_train)
            train_errors.append(train_error)
            
            #validation error
            ytrue_val = val_dataset.Y
            val_mask = getattr(val_dataset, 'mask', None)
            ypred_val = model(val_dataset.X, mask=val_mask)
            val_error = loss_output(ytrue_val, ypred_val)
            val_errors.append(val_error)

        # Step the LR scheduler based on validation error
        scheduler.step(val_error)

        #logging with tqdm
        if verbose:
            pbar.set_postfix({
                "Train Error": formatter(train_errors[-1]), 
                "Val Error": formatter(val_errors[-1]),
                "Patience": patience_counter,
                "Best Val Error": formatter(torch.min(torch.tensor(val_errors))),
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

        #early stopping
        if val_error == torch.min(torch.tensor(val_errors)):    #if the current validation error is the best one so far
            best_params = model.state_dict()    # Save the best model parameters
            patience_counter = 0                # Reset the patience counter
        else:
            patience_counter += 1                # Increment the patience counter

        #stop training if the patience counter exceeds the specified patience
        if patience_counter >= patience:      
            model.load_state_dict(best_params)  # Load the best model parameters before returning 
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
                pbar.close()

            train_err_np = torch.tensor(train_errors).detach().cpu().numpy()
            val_err_np = torch.tensor(val_errors).detach().cpu().numpy()
            return train_err_np, val_err_np

    # If we finish all epochs without triggering early stopping
    model.load_state_dict(best_params)  # Load the best model parameters before returning
    if verbose:
        print(f"Training Complete ({epochs} epochs)")
        pbar.close()
    train_err_np = torch.tensor(train_errors).detach().cpu().numpy()
    val_err_np = torch.tensor(val_errors).detach().cpu().numpy()
    return train_err_np, val_err_np
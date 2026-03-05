import torch            #type: ignore
import torch.nn as nn   #type: ignore
from torch.utils.data import DataLoader  #type: ignore
import numpy as np      #type: ignore
from types import SimpleNamespace
from tqdm import tqdm   #type: ignore

from .transformer import Transformer
from .sdn import MLP, CNN
from .utils import num2p, mre, mse

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
    patience: int = 100
):
    #init the data, the optimizer, params ...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer(model.parameters(), lr=lr)
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
                output = model(data[0])               #forward pass
                loss = loss_fn(data[1], output["output"].squeeze(1))  #compute the loss
                loss.backward()                       #backward pass
                return loss
            optimizer.step(closure)                 #optimizer step

        #compute the training and validation errors
        model.eval()                            #set the model to evaluation mode
        with torch.no_grad():                   #disable gradient computation
            #train error
            ytrue_train = train_dataset.Y
            ypred_train = model(train_dataset.X)
            train_error = loss_output(ytrue_train, ypred_train["output"].squeeze(1))
            train_errors.append(train_error)
            #validation error
            ytrue_val = val_dataset.Y
            ypred_val = model(val_dataset.X)
            val_error = loss_output(ytrue_val, ypred_val["output"].squeeze(1))
            val_errors.append(val_error)

        #logging with tqdm
        if verbose:
            pbar.set_postfix({
                "Train Error": formatter(train_errors[-1]), 
                "Val Error": formatter(val_errors[-1]),
                "Patience": patience_counter,
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
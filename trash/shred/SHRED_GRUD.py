import torch                            # type: ignore
import torch.nn as nn                   # type: ignore
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm                   # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau      # type: ignore

from ..old.utils import mae, mse, mre, num2p
from .GRUD import GRUD
from .AttnGRUD import AttnGRUD

class SHRED_GRUD(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        X_mean: torch.Tensor,
        hidden_size: int = 64,
        decoder_sizes: list[int] = [350,400],
        dropout: float = 0.0,
        attention: bool = False
    ):
        super().__init__()

        if not attention:
            #GRU-D encoder (replaces the LSTM encoder from classic SHRED)
            self.encoder = GRUD(
                input_size = input_size,
                cell_size = hidden_size,
                hidden_size = hidden_size,
                X_mean = X_mean,
                output_last = True
            )
        else:
            #AttnGRU-D encoder (replaces the LSTM encoder from classic SHRED)
            self.encoder = AttnGRUD(
                input_size = input_size,
                cell_size = hidden_size,
                hidden_size = hidden_size,
                X_mean = X_mean,
                output_last = True
            )
        
        #MLP decoder
        self.decoder = nn.ModuleList()
        decoder_sizes = [hidden_size] + decoder_sizes + [output_size]
        # Create linear layers for the decoder
        for i in range(len(decoder_sizes)-1):                                       # For each layer in the decoder
            self.decoder.append(nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))    # Add a linear layer from the current size to the next size
            if i < len(decoder_sizes)-2:                                            # Add ReLU and dropout activation for all layers except the last one
                self.decoder.append(nn.ReLU())                          
                if dropout > 0.0:
                    self.decoder.append(nn.Dropout(dropout))

        #store the needed parameters for the forward pass
        self.hidden_size = hidden_size

    def forward(self, x):
        # x must be shaped (batch_size, 4, seq_len, input_size)
        # The 4 channels are: X, X_last_obsv, Mask, Delta
        # GRUD manages its own hidden-state initialisation internally
        out = self.encoder(x)       # (batch_size, hidden_size)
        for layer in self.decoder:  # pass through the MLP decoder
            out = layer(out)

        # Return the final output of the model
        return out

    def freeze(self):
        self.eval()  # Set the model to evaluation mode
        for param in self.parameters():
            param.requires_grad = False  # Freeze the parameters by setting requires_grad to False

    def unfreeze(self):
        self.train()  # Set the model to training mode
        for param in self.parameters():
            param.requires_grad = True  # Unfreeze the parameters by setting requires_grad to True

    
def fit(
    model: SHRED_GRUD,
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
                output = model(data[0])               #forward pass
                loss = loss_fn(data[1], output)       #compute the loss
                loss.backward()                       #backward pass
                return loss
            optimizer.step(closure)                 #optimizer step

        #compute the training and validation errors
        model.eval()                            #set the model to evaluation mode
        with torch.no_grad():                   #disable gradient computation
            #train error
            ytrue_train = train_dataset.Y
            ypred_train = model(train_dataset.X)
            train_error = loss_output(ytrue_train, ypred_train)
            train_errors.append(train_error)
            #validation error
            ytrue_val = val_dataset.Y
            ypred_val = model(val_dataset.X)
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

def predict(
    model: SHRED_GRUD, 
    input_data: torch.Tensor,
    steps: int,
    nsensors: int
):
    # Predict future values using the trained SHRED model
    forecast = []
    for _ in range(steps):
        forecast.append(model(input_data))          # Predict the next time step using the current input data
        temp = input_data.clone()                   # Create a copy of the input data to modify for the next prediction
        input_data[:,:-1] = temp[:,1:]              # Shift the input data to the left by one time step
        input_data[:,-1,:nsensors] = forecast[-1]   # Set the last time step of the input data to the predicted values for the next prediction

    # Stack the list of forecasts into a single tensor and return it
    return torch.stack(forecast, dim=1)           
            



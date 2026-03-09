import torch                            # type: ignore
import torch.nn as nn                   # type: ignore
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm                   # type: ignore

from ..utils import mae, mse, mre, num2p

class SHRED(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        decoder_sizes: list[int] = [350,400],
        dropout: float = 0.0
    ):
        super().__init__()

        #LSTM encoder
        self.encoder = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = hidden_layers,
            batch_first = True)
        
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
        self.hidden_layers = hidden_layers

    def forward(self, x):
        # init of h0 and c0 for the LSTM
        h0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)

        # Move h0 and c0 to the same device as the model parameters
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # SHRED forward pass
        _, (hn, _) = self.encoder(x, (h0, c0))  # Pass the input through the LSTM encoder
        out = hn[-1].view(-1, self.hidden_size) # Take the last layer's hidden state as the output
        for layer in self.decoder:              # Pass the output through the decoder layers
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
    model: SHRED,
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

def predict(
    model: SHRED, 
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
            



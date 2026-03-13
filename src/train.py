from typing import Union, List, Tuple, Optional #type: ignore
import torch                                    #type: ignore
from torch.utils.data import DataLoader         #type: ignore
from tqdm import tqdm                           #type: ignore

from .shred.SHRED import SHRED
from .shred.SHRED_GRUD import SHRED_GRUD
from .tshred.TSHRED import TSHRED


ModelType = Union[SHRED, SHRED_GRUD, TSHRED]

def fit(
    model: ModelType,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    num_epochs: int = 200,
    lr: float = 1e-3,
    optimizer: str = 'AdamW',
    weight_decay: float = 0.01,
    patience: int = 20,
    verbose: bool = True
):
    
    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train loader 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # criterion and optimizer
    criterion = torch.nn.MSELoss()
    val_criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = _get_optimizer(model.parameters(), optimizer, lr, weight_decay)

    # error lists
    train_losses = []
    val_losses = []

    # early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_params = model.state_dict()

    # start training loop
    print("Starting training...")
    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    for epoch in pbar:

        #train
        model.train()
        train_loss = 0.0
        for X, Y in train_loader:
            outputs = model(X)              #forward pass
            optimizer.zero_grad()           #zero the gradients
            loss = criterion(outputs, Y)    #compute the loss
            loss.backward()                 #backward pass
            optimizer.step()                #optimizer step
            train_loss += loss.item()       # accumulate the loss

        train_loss = train_loss / len(train_loader)  # average loss over batches
        train_losses.append(train_loss)

        #evaluate on training and validation sets
        model.eval()
        with torch.no_grad():

            #val error
            val_loss = 0.0
            val_elements = 0
            for X, Y in val_loader:
                outputs = model(X)                  #forward pass
                loss = val_criterion(outputs, Y)    #compute the loss
                val_loss += loss.item()             # accumulate the loss
                val_elements += Y.numel()           # accumulate the number of elements

            val_loss = val_loss / val_elements  # average loss over all elements
            val_losses.append(val_loss)

        #logging with tqdm
        if verbose:
            pbar.set_postfix({
                "Train Loss": f"{train_loss:.4f}", 
                "Val Loss": f"{val_loss:.4f}",
                "Patience": patience_counter,
                "Best Val Loss": f"{best_val_loss:.4f}"
                })
    
        #early stopping
        # If the validation loss has improved, save the model parameters and reset the patience counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = model.state_dict()
            patience_counter = 0
        # If the validation loss has not improved, increment the patience counter
        else:
            patience_counter += 1
        # If the patience counter exceeds the specified patience, stop training and load the best model parameters
        if patience_counter >= patience:
            model.load_state_dict(best_params)
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    #End of training loop
    model.load_state_dict(best_params)  # Load the best model parameters before returning
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")
    pbar.close()

    # Convert the lists of training and validation losses to numpy arrays before returning
    train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
    val_losses_np = torch.tensor(val_losses).detach().cpu().numpy()
    return train_losses_np, val_losses_np


def evaluate(
    model: ModelType,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    loss_fn = torch.nn.MSELoss(reduction='sum')
):
    
    #evaluate the model on the given dataset and return the error
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_elements = 0

    with torch.no_grad():
        for X, Y in dataloader:
            outputs = model(X)                  #forward pass
            loss = loss_fn(outputs, Y)          #compute the loss
            total_loss += loss.item()           # accumulate the loss
            total_elements += Y.numel()         # accumulate the number of elements

    # compute the mean loss over all elements
    mse = total_loss / total_elements
    return mse

def _get_optimizer(params, optimizer: str, lr: float, weight_decay: float):
    if optimizer == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Supported optimizers are: AdamW, Adam, SGD, RMSprop, Adagrad."
        )
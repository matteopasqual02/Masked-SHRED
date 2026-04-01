from typing import Union, List, Tuple, Optional #type: ignore
import copy                                     #type: ignore
import torch                                    #type: ignore
from torch.utils.data import DataLoader         #type: ignore
from tqdm import tqdm                           #type: ignore

from .mSHRED.mSRED import mSHRED


def fit(
    model: mSHRED,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    batch_size: int = 64,
    num_epochs: int = 200,
    lr: float = 1e-3,
    optimizer: str = 'AdamW',
    lr_scheduler: Optional[str] = None,
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.2,
    scheduler_embed_dim: Optional[int] = None,
    scheduler_warmup_steps: Optional[int] = None,
    weight_decay: float = 0.005,
    patience: int = 20,
    verbose: bool = True,
    masking_type: Optional[str] = None,
):
    
    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train loader 
    has_val = val_dataset is not None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if has_val:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # criterion, optimizer, and scheduler
    criterion = torch.nn.MSELoss()          # used for backward pass
    val_criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = _get_optimizer(model.parameters(), optimizer, lr, weight_decay)
    scheduler = None
    if lr_scheduler is not None:
        scheduler = _get_lr_scheduler(
            optimizer=optimizer,
            scheduler=lr_scheduler,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma,
            embed_dim=scheduler_embed_dim,
            warmup_steps=scheduler_warmup_steps,
        )

    # error lists
    train_losses = []
    val_losses = []

    if has_val:
        # early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = copy.deepcopy(model.state_dict())  # deep copy to avoid reference issues

    # start training loop
    print("Starting training...")
    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    is_mask_data = masking_type == 'mask_data'
    for epoch in pbar:

        #train
        model.train()
        # If the dataset includes masks, we need to unpack the batches differently and pass the masks to the model.
        if is_mask_data:
            for batch in train_loader:
                X, M, Y = batch
                outputs = model(X, M)       # forward pass with explicit data mask
                optimizer.zero_grad()           #zero the gradients
                loss = criterion(outputs, Y)    #compute the loss
                loss.backward()                 #backward pass
                optimizer.step()                #optimizer step

                # transformer scheduler step (if applicable, step once per optimizer step)
                # NOTE: for the "Transformer" scheduler, we step once per optimizer step, 
                # so we only step here if it's the "Transformer" scheduler
                if scheduler is not None and lr_scheduler == "Transformer":
                    scheduler.step()
        else:
            for batch in train_loader:
                X, Y = batch
                outputs = model(X)          # forward pass
                optimizer.zero_grad()           #zero the gradients
                loss = criterion(outputs, Y)    #compute the loss
                loss.backward()                 #backward pass
                optimizer.step()                #optimizer step

                # transformer scheduler step (if applicable, step once per optimizer step)
                # NOTE: for the "Transformer" scheduler, we step once per optimizer step, 
                # so we only step here if it's the "Transformer" scheduler
                if scheduler is not None and lr_scheduler == "Transformer":
                    scheduler.step()

        #evaluate on training and validation sets (both in eval mode for consistency)
        model.eval()
        with torch.no_grad():

            #train error
            train_loss = 0.0
            train_elements = 0
            # If the dataset includes masks, we need to unpack the batches differently and pass the masks to the model.
            if is_mask_data:
                for batch in train_loader:
                    X, M, Y = batch
                    outputs = model(X, M)           # forward pass with explicit data mask
                    loss = val_criterion(outputs, Y)    #compute the loss
                    train_loss += loss.item()           # accumulate the loss
                    train_elements += Y.numel()         # accumulate the number of elements
            else:
                for batch in train_loader:
                    X, Y = batch
                    outputs = model(X)              # forward pass
                    loss = val_criterion(outputs, Y)    #compute the loss
                    train_loss += loss.item()           # accumulate the loss
                    train_elements += Y.numel()         # accumulate the number of elements
                
            train_loss = train_loss / train_elements  # average loss over all elements
            train_losses.append(train_loss)

            if has_val:
                #val error
                val_loss = 0.0
                val_elements = 0
                # If the dataset includes masks, we need to unpack the batches differently and pass the masks to the model.
                if is_mask_data:
                    for batch in val_loader:
                        X, M, Y = batch
                        outputs = model(X, M)           # forward pass with explicit data mask
                        loss = val_criterion(outputs, Y)    #compute the loss
                        val_loss += loss.item()             # accumulate the loss
                        val_elements += Y.numel()           # accumulate the number of elements
                else:
                    for batch in val_loader:
                        X, Y = batch
                        outputs = model(X)              # forward pass
                        loss = val_criterion(outputs, Y)    #compute the loss
                        val_loss += loss.item()             # accumulate the loss
                        val_elements += Y.numel()           # accumulate the number of elements

                val_loss = val_loss / val_elements  # average loss over all elements
                val_losses.append(val_loss)

        # step the learning rate scheduler 
        # NOTE: for the "Transformer" scheduler, we step once per optimizer step, 
        # so we only step here if it's not the "Transformer" scheduler
        if scheduler is not None and lr_scheduler != "Transformer":
            scheduler.step()

        #logging with tqdm
        if verbose:
            if has_val:
                pbar.set_postfix({
                    "Train Loss": f"{train_loss:.4f}", 
                    "Val Loss": f"{val_loss:.4f}",
                    "Patience": patience_counter,
                    "Best Val Loss": f"{best_val_loss:.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
            else:
                pbar.set_postfix({
                    "Train Loss": f"{train_loss:.4f}", 
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
    
        if has_val:
            #early stopping
            # If the validation loss has improved, save the model parameters and reset the patience counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = copy.deepcopy(model.state_dict())  # deep copy to avoid reference issues
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
    if has_val:
        model.load_state_dict(best_params)  # Load the best model parameters before returning
        print(f"Training completed with best validation loss: {best_val_loss:.4f}")
    else:
        print("Training completed without validation set (early stopping disabled).")
    pbar.close()

    # Convert the lists of training and validation losses to numpy arrays before returning
    train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
    val_losses_np = torch.tensor(val_losses).detach().cpu().numpy()
    return train_losses_np, val_losses_np

def evaluate(
    model: mSHRED,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    loss_fn = torch.nn.MSELoss(reduction='sum'),
    eps: float = 1e-12,
    masking_type: Optional[str] = None,
):
    
    #evaluate the model on the given dataset and return the error
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_elements = 0
    total_mre = 0.0
    total_mre_elements = 0
    is_mask_data = masking_type == 'mask_data'
    
    with torch.no_grad():
        # If the dataset includes masks, we need to unpack the batches differently and pass the masks to the model.
        if is_mask_data:
            for X, M, Y in dataloader:
                outputs = model(X, M)               # forward pass with explicit data mask
                loss = loss_fn(outputs, Y)          #compute the loss
                total_loss += loss.item()           # accumulate the loss
                total_elements += Y.numel()         # accumulate the number of elements

                # Mean Relative Error over the last axis, averaged over all remaining axes.
                numerator = (Y - outputs).pow(2).sum(dim=-1).sqrt()         # compute the numerator of the relative error
                denominator = Y.pow(2).sum(dim=-1).sqrt().clamp_min(eps)    # compute the denominator of the relative error, with a small epsilon to avoid division by zero
                rel_error = numerator / denominator                         # compute the relative error
                total_mre += rel_error.sum().item()                         # accumulate the relative error
                total_mre_elements += rel_error.numel()                     # accumulate the number of elements for the relative error
        else:
            for X, Y in dataloader:
                outputs = model(X)                  #forward pass
                loss = loss_fn(outputs, Y)          #compute the loss
                total_loss += loss.item()           # accumulate the loss
                total_elements += Y.numel()         # accumulate the number of elements

                # Mean Relative Error over the last axis, averaged over all remaining axes.
                numerator = (Y - outputs).pow(2).sum(dim=-1).sqrt()         # compute the numerator of the relative error
                denominator = Y.pow(2).sum(dim=-1).sqrt().clamp_min(eps)    # compute the denominator of the relative error, with a small epsilon to avoid division by zero
                rel_error = numerator / denominator                         # compute the relative error
                total_mre += rel_error.sum().item()                         # accumulate the relative error
                total_mre_elements += rel_error.numel()                     # accumulate the number of elements for the relative error

    # compute the mean loss over all elements
    mse = total_loss / total_elements
    mre = total_mre / total_mre_elements

    return mse, mre

def mre(datatrue, datapred):
    # Mean Relative Error
    return ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() 
        / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()

def mse(datatrue, datapred):
    # Mean Squared Error
    return (datatrue - datapred).pow(2).sum(axis=-1).mean() / datatrue.shape[-1]

def fit2(
    model: mSHRED,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    batch_size: int = 64,
    num_epochs: int = 200,
    lr: float = 1e-3,
    optimizer: str = 'AdamW',
    lr_scheduler: Optional[str] = None,
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.2,
    scheduler_embed_dim: Optional[int] = None,
    scheduler_warmup_steps: Optional[int] = None,
    weight_decay: float = 0.005,
    patience: int = 20,
    verbose: bool = True,
    masking_type: Optional[str] = None,
):
    
    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train loader 
    has_val = val_dataset is not None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # criterion, optimizer, and scheduler
    criterion = torch.nn.MSELoss()          # used for backward pass
    optimizer = _get_optimizer(model.parameters(), optimizer, lr, weight_decay)
    scheduler = None
    if lr_scheduler is not None:
        scheduler = _get_lr_scheduler(
            optimizer=optimizer,
            scheduler=lr_scheduler,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma,
            embed_dim=scheduler_embed_dim,
            warmup_steps=scheduler_warmup_steps,
        )

    # error lists
    train_losses = []
    val_losses = []

    if has_val:
        # early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = copy.deepcopy(model.state_dict())  # deep copy to avoid reference issues

    # start training loop
    print("Starting training...")
    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    is_mask_data = masking_type == 'mask_data'
    for epoch in pbar:

        #train
        model.train()
        # If the dataset includes masks, we need to unpack the batches differently and pass the masks to the model.
        if is_mask_data:
            for batch in train_loader:
                X, M, Y = batch
                outputs = model(X, M)       # forward pass with explicit data mask
                optimizer.zero_grad()           #zero the gradients
                loss = criterion(outputs, Y)    #compute the loss
                loss.backward()                 #backward pass
                optimizer.step()                #optimizer step

                # transformer scheduler step (if applicable, step once per optimizer step)
                # NOTE: for the "Transformer" scheduler, we step once per optimizer step, 
                # so we only step here if it's the "Transformer" scheduler
                if scheduler is not None and lr_scheduler == "Transformer":
                    scheduler.step()
        else:
            for batch in train_loader:
                X, Y = batch
                outputs = model(X)          # forward pass
                optimizer.zero_grad()           #zero the gradients
                loss = criterion(outputs, Y)    #compute the loss
                loss.backward()                 #backward pass
                optimizer.step()                #optimizer step

                # transformer scheduler step (if applicable, step once per optimizer step)
                # NOTE: for the "Transformer" scheduler, we step once per optimizer step, 
                # so we only step here if it's the "Transformer" scheduler
                if scheduler is not None and lr_scheduler == "Transformer":
                    scheduler.step()

        #evaluate on training and validation sets (both in eval mode for consistency)
        model.eval()
        with torch.no_grad():

            # train error
            if is_mask_data:
                ytrue_train = train_dataset.Y
                ypred_train = model(train_dataset.X, train_dataset.M)
                train_loss = mse(ytrue_train, ypred_train)
                train_losses.append(train_loss)
            else:
                ytrue_train = train_dataset.Y
                ypred_train = model(train_dataset.X)
                train_loss = mse(ytrue_train, ypred_train)
                train_losses.append(train_loss)

            if has_val:
                #val error
                if is_mask_data:
                    ytrue_val = val_dataset.Y
                    ypred_val = model(val_dataset.X, val_dataset.M)
                    val_loss = mse(ytrue_val, ypred_val)
                    val_losses.append(val_loss)
                else:
                    ytrue_val = val_dataset.Y
                    ypred_val = model(val_dataset.X)
                    val_loss = mse(ytrue_val, ypred_val)
                    val_losses.append(val_loss)
                    

        # step the learning rate scheduler 
        # NOTE: for the "Transformer" scheduler, we step once per optimizer step, 
        # so we only step here if it's not the "Transformer" scheduler
        if scheduler is not None and lr_scheduler != "Transformer":
            scheduler.step()

        #logging with tqdm
        if verbose:
            if has_val:
                pbar.set_postfix({
                    "Train Loss": f"{train_loss:.4f}", 
                    "Val Loss": f"{val_loss:.4f}",
                    "Patience": patience_counter,
                    "Best Val Loss": f"{best_val_loss:.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
            else:
                pbar.set_postfix({
                    "Train Loss": f"{train_loss:.4f}", 
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
    
        if has_val:
            #early stopping
            # If the validation loss has improved, save the model parameters and reset the patience counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = copy.deepcopy(model.state_dict())  # deep copy to avoid reference issues
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
    if has_val:
        model.load_state_dict(best_params)  # Load the best model parameters before returning
        print(f"Training completed with best validation loss: {best_val_loss:.4f}")
    else:
        print("Training completed without validation set (early stopping disabled).")
    pbar.close()

    # Convert the lists of training and validation losses to numpy arrays before returning
    train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
    val_losses_np = torch.tensor(val_losses).detach().cpu().numpy()
    return train_losses_np, val_losses_np


def evaluate2(
    model: mSHRED,
    dataset: torch.utils.data.Dataset,
    masking_type: Optional[str] = None,
):
    
    #evaluate the model on the given dataset and return the error
    model.eval()
    is_mask_data = masking_type == 'mask_data'
    with torch.no_grad():
        if is_mask_data:
            ytrue = dataset.Y
            ypred = model(dataset.X, dataset.M)
            e_mre = mre(ytrue, ypred)
            e_mse = mse(ytrue, ypred)
        else:
            ytrue = dataset.Y
            ypred = model(dataset.X)
            e_mre = mre(ytrue, ypred)
            e_mse = mse(ytrue, ypred)

    return e_mse, e_mre


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
    
def _get_lr_scheduler(
    optimizer,
    scheduler: str,
    step_size: int = 10,
    gamma: float = 0.1,
    embed_dim: Optional[int] = None,
    warmup_steps: Optional[int] = None,
):
    
    if scheduler in {"Classical", "StepLR"}:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    
    elif scheduler == "Transformer":
        if embed_dim is None or warmup_steps is None:
            raise ValueError("'Transformer' scheduler requires 'embed_dim' and 'warmup_steps'.")
        if warmup_steps <= 0:
            raise ValueError("'warmup_steps' must be > 0 for Transformer scheduler.")

        def warmup_lr_lambda(step: int) -> float:
            step = max(step, 1)
            return (embed_dim ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

        # Step-based schedule: call scheduler.step() once per optimizer step.
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    
    else:
        raise ValueError("Unsupported scheduler: {scheduler}. ")
## This file also contains a utility function for plotting training curves: 
# - plot_training_curves: plot the training and validation MSE loss curves over epochs using matplotlib
import mikeio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import modelskill as ms
import torch

import sys
from pathlib import Path
import os

def plot_training_curves(train_errors, valid_errors):

    # PLOT TRAINING CURVES
    plt.figure(figsize=(10, 5))
    plt.plot(train_errors, label='Train MSE')
    plt.plot(valid_errors, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation MSE Loss Curves')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_model(
    evaluate,
    mshred,
    train_dataset,
    val_dataset,
    test_dataset,
    model,
    type,
    masking_type=None,
):
    
    # MSE and MRE
    if masking_type is None:
        train_mse, train_mre = evaluate(mshred, train_dataset)
        val_mse, val_mre = evaluate(mshred, val_dataset)
        test_mse, test_mre = evaluate(mshred, test_dataset)
    else:
        train_mse, train_mre = evaluate(mshred, train_dataset, masking_type=masking_type)
        val_mse, val_mre = evaluate(mshred, val_dataset, masking_type=masking_type)
        test_mse, test_mre = evaluate(mshred, test_dataset, masking_type=masking_type)
        
    test_error = {
        "model": model,
        "type": type,
        "train-mse": float(train_mse),
        "val-mse": float(val_mse),
        "test-mse": float(test_mse),
        "train-mre": float(train_mre),
        "val-mre": float(val_mre),
        "test-mre": float(test_mre),
    }
    return test_error


def plot_reconstruction(test_reconstruction, X_raw, sst_locs, index=-1):
    #compare the last step of the reconstruction with the original data
    last_step_hat = test_reconstruction['Surface elevation'][index]
    last_step_hat_raw = np.zeros(X_raw.shape[1])
    last_step_hat_raw[sst_locs] = last_step_hat

    X_raw_last_step = X_raw[index]

    import copy

    X_orig = X_raw_last_step.reshape(180, 360)
    X_hat = last_step_hat_raw.reshape(180, 360)
    X_err = np.abs(X_orig - X_hat)

    # Mask land — adjust the condition to match how land is actually encoded
    # (NaN is most common; swap for `== 0` or another sentinel if that's your case)
    land_mask = np.isnan(X_orig)

    X_orig_m = np.ma.masked_where(land_mask, X_orig)
    X_hat_m  = np.ma.masked_where(land_mask, X_hat)
    X_err_m  = np.ma.masked_where(land_mask, X_err)

    vmin = min(X_orig_m.min(), X_hat_m.min())
    vmax = max(X_orig_m.max(), X_hat_m.max())

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
    })

    # Black land for the two temperature maps; error map keeps its own cmap as-is
    cmap_temp = copy.copy(plt.get_cmap('RdYlBu_r'))
    cmap_temp.set_bad(color='black')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    panels = [
        (X_orig_m, 'Ground Truth', cmap_temp, vmin, vmax, '°C'),
        (X_hat_m,  'M-SHRED',      cmap_temp, vmin, vmax, '°C'),
        (X_err_m,  'Absolute Error', 'inferno', None, None, '°C'),
    ]

    for ax, (data, title, cmap, vmn, vmx, unit) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmn, vmax=vmx,
                        aspect='equal', origin='upper')
        ax.set_title(title, fontweight='semibold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(unit, fontsize=9, labelpad=4)
        cbar.ax.tick_params(labelsize=8, length=3)
        cbar.outline.set_linewidth(0.6)

    plt.tight_layout()

    plt.savefig('outputs/sst_reconstruction.pdf', bbox_inches='tight')
    plt.savefig('outputs/sst_reconstruction.png', dpi=300, bbox_inches='tight')

    plt.show()
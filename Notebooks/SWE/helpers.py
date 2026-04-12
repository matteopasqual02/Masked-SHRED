import numpy as np                  #type: ignore
import pandas as pd                 #type: ignore
import matplotlib.pyplot as plt     #type: ignore
import torch                        #type: ignore

from src.datamanager import DataManager
from src.engine import SHREDEngine
from src.train import fit2, evaluate2
from src.mSHRED.mSRED import mSHRED


# Utils functions
def to_numpy(x):
    """Return a NumPy array from a torch tensor or array-like input.""" 

    # Handle torch tensors explicitly to move data to CPU and detach from graph.
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def compute_errors_previous_method(target, prediction, eps=1e-12):
    """Compute MSE and MRE with the previous axis-wise relative-error method."""

    # Pointwise error used for MSE.
    error = target - prediction
    mse = np.nanmean(error ** 2)

    # Mean Relative Error: ||target - prediction|| / ||target|| over last axis,
    # then averaged over all remaining axes.
    numerator = np.sqrt(np.nansum((target - prediction) ** 2, axis=-1))
    denominator = np.sqrt(np.nansum(target ** 2, axis=-1))
    rel_error = numerator / np.maximum(denominator, eps)
    mre = np.nanmean(rel_error)

    return float(mse), float(mre)

# Function to visualize sensor locations on the SWE grid
def plot_sensors_on_field(H, Lx, Ly, NSENSORS, sensor_rc, time_idx=0):
    """Plot sensor positions over the SWE height field at a selected time index."""

    # Plot one spatial snapshot of the height field.
    plt.figure(figsize=(6, 4))
    im = plt.imshow(H[time_idx].reshape(Lx, Ly), cmap="gist_earth", origin="lower")
    plt.colorbar(im, shrink=0.8)

    # Overlay each sensor position and numeric label.
    for k in range(NSENSORS):
        r, c = sensor_rc[k]
        plt.plot(c, r, "o", mfc="magenta", mec="black", ms=7, mew=1.0)
        plt.text(c + 1, r + 1, str(k), color="white", fontsize=8)
    plt.title(f"Sensors on SWE grid (t={time_idx})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Function to build the DataManager and prepare datasets
def build_manager(
    H,
    Utheta,
    Uphi,
    LAGS,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    COMPRESS,
    measurements_input,
    imputer=None,
    masking_type=None,
):
    """Create DataManager, register fields, and return prepared train/val/test splits."""

    # Initialize DataManager with temporal lag and split ratios.
    manager = DataManager(
        lags=LAGS,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
    )

    # Register target field with sensor measurements and optional imputer.
    manager.add_data(H, id="H", compress=COMPRESS, measurements=measurements_input, imputer=imputer)
    # Register auxiliary fields used by the multi-field reconstruction setup.
    manager.add_data(Utheta, id="Utheta", compress=COMPRESS)
    manager.add_data(Uphi, id="Uphi", compress=COMPRESS)

    # GRUD preparation returns X_mean, which is required by GRUD variants.
    if masking_type == "grud":
        train_dataset, val_dataset, test_dataset, X_mean = manager.prepare(verbose=True, masking_type="grud")
        return manager, train_dataset, val_dataset, test_dataset, X_mean

    # Other masking modes use the standard dataset tuple without X_mean.
    train_dataset, val_dataset, test_dataset = manager.prepare(verbose=True, masking_type=masking_type)
    return manager, train_dataset, val_dataset, test_dataset, None

# Function to create model arguments based on the encoder type
def make_model_args(
    encoder,
    nn_input_shape,
    nn_output_shape,
    LAGS,
    TR_ENCODER_DEPTH,
    TR_N_HEADS,
    TR_DIM_FEEDFORWARD,
    TR_HIDDEN_SIZE,
    TR_DROPOUT,
    LSTM_NUM_LAYERS,
    GRUD_HIDDEN_SIZE,
    GRUD_DROPOUT,
    X_mean=None,
    double_channels=False,
):
    """Build the mSHRED model argument dictionary for the selected encoder."""

    # Transformer-family encoders share the same base hyperparameter schema.
    if encoder in ["transformer", "transformer_hidden_mask", "transformer_attn_mask", "transformer_learned_mask"]:
        return dict(
            encoder=encoder,
            encoder_depth=TR_ENCODER_DEPTH,
            input_size=nn_input_shape,
            n_heads=TR_N_HEADS,
            dim_feedforward=TR_DIM_FEEDFORWARD,
            hidden_size=TR_HIDDEN_SIZE,
            input_length=LAGS,
            double_channels=double_channels,
            decoder="mlp",
            decoder_hidden_sizes=[],
            output_size=nn_output_shape,
            dropout=TR_DROPOUT,
        )
    # LSTM encoder-specific arguments.
    if encoder == "lstm":
        return dict(
            encoder="lstm",
            input_size=nn_input_shape,
            hidden_size=TR_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            decoder="mlp",
            decoder_hidden_sizes=[],
            output_size=nn_output_shape,
            dropout=TR_DROPOUT,
        )
    # GRUD and attention-GRUD need feature mean values for decay/imputation behavior.
    if encoder in ["grud", "att_grud"]:
        return dict(
            encoder=encoder,
            input_size=nn_input_shape,
            hidden_size=GRUD_HIDDEN_SIZE,
            X_mean=X_mean,
            decoder="mlp",
            decoder_hidden_sizes=[],
            output_size=nn_output_shape,
            dropout=GRUD_DROPOUT,
        )
    # Fail fast on unknown encoder names.
    raise ValueError(f"Unsupported encoder: {encoder}")

# Function to plot the true vs reconstructed SWE field and the error
def plot_swe_reconstruction(manager, test_dataset, model, label, H, Utheta, Uphi, Lx, Ly):
    """Plot true/reconstructed/error maps for H, Utheta, and Uphi with per-field MSE and MRE."""

    # Build inference engine from trained model and manager metadata.
    engine = SHREDEngine(manager, model)
    # Reconstruct fields from test sensor sequences.
    rec = engine.sensor_to_reconstruction_dataset(test_dataset)
    test_global_idx = np.asarray(manager.test_indices)

    # Use a representative middle test snapshot when available.
    ntest = len(test_global_idx)
    test_local_idx = min(ntest // 2, max(0, ntest - 1))
    t_global = test_global_idx[test_local_idx]

    # Field configuration: source arrays and colormaps for plotting.
    fields = [
        ("H", H, "gist_earth"),
        ("Utheta", Utheta, "twilight_shifted"),
        ("Uphi", Uphi, "twilight_shifted"),
    ]

    # Compute per-field metrics using the previous method.
    metrics = {}
    for field_name, field_true_all, _ in fields:
        target = field_true_all[test_global_idx]
        pred = rec[field_name]
        mse, mre = compute_errors_previous_method(target, pred)
        metrics[field_name] = {"mse": mse, "mre": mre}

    # Render one row per field and three columns: true, reconstructed, absolute error.
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    for row, (field_name, field_true_all, cmap) in enumerate(fields):
        field_pred_all = rec[field_name]
        mse_field = float(metrics[field_name]["mse"])
        mre_field = float(metrics[field_name]["mre"])

        # Snapshot values used in map visualization.
        field_true = field_true_all[t_global]
        field_hat = field_pred_all[test_local_idx]
        field_err = np.abs(field_true - field_hat)
        vmin, vmax = field_true.min(), field_true.max()

        im_true = axes[row, 0].imshow(
            field_true.reshape(Lx, Ly),
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[row, 0].set_title(f"{field_name} true")
        axes[row, 0].axis("off")
        plt.colorbar(im_true, ax=axes[row, 0], shrink=0.8)

        im_hat = axes[row, 1].imshow(
            field_hat.reshape(Lx, Ly),
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[row, 1].set_title(
            f"{field_name} reconstructed\nMSE={mse_field:.6f}, MRE={mre_field:.6f}"
        )
        axes[row, 1].axis("off")
        plt.colorbar(im_hat, ax=axes[row, 1], shrink=0.8)

        im_err = axes[row, 2].imshow(field_err.reshape(Lx, Ly), cmap="Reds", origin="lower")
        axes[row, 2].set_title(f"{field_name} |error|")
        axes[row, 2].axis("off")
        plt.colorbar(im_err, ax=axes[row, 2], shrink=0.8)

    plt.suptitle(f"{label}: SWE reconstruction (H, Utheta, Uphi)")
    plt.tight_layout()
    plt.show()

    # Return engine so caller can explicitly delete it if desired.
    return engine

# Main function to run the experiment with the specified configuration
def run_experiment(
    label,
    exp_type,
    measurements_input,
    encoder,
    H,
    Utheta,
    Uphi,
    device,
    ERRORS,
    Lx,
    Ly,
    LAGS,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    COMPRESS,
    OPTIMIZER,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    PATIENCE,
    VERBOSE_TRAIN,
    LR_SCHEDULER,
    SCHEDULER_STEP_SIZE,
    TR_ENCODER_DEPTH,
    TR_N_HEADS,
    TR_DIM_FEEDFORWARD,
    TR_HIDDEN_SIZE,
    TR_DROPOUT,
    LSTM_NUM_LAYERS,
    GRUD_HIDDEN_SIZE,
    GRUD_DROPOUT,
    imputer=None,
    masking_type=None,
    double_channels=False,
    make_plot=True,
):
    """Train and evaluate one experiment, log metrics, and optionally plot reconstruction."""

    # Prepare DataManager datasets for the selected imputation/masking mode.
    manager, train_dataset, val_dataset, test_dataset, X_mean = build_manager(
        H=H,
        Utheta=Utheta,
        Uphi=Uphi,
        LAGS=LAGS,
        TRAIN_SIZE=TRAIN_SIZE,
        VAL_SIZE=VAL_SIZE,
        TEST_SIZE=TEST_SIZE,
        COMPRESS=COMPRESS,
        measurements_input=measurements_input,
        imputer=imputer,
        masking_type=masking_type,
    )

    # Visual separator in notebook output.
    print("\n", "-" * 100, "\n")

    # Infer input/output dimensions from dataset tuple layout for each masking mode.
    if masking_type == "grud":
        nn_input_shape = train_dataset[0][0].shape[2]
        nn_output_shape = train_dataset[0][1].shape[0]
    elif masking_type == "mask_data":
        nn_input_shape = train_dataset[0][0].shape[1]
        nn_output_shape = train_dataset[0][2].shape[0]
    else:
        nn_input_shape = train_dataset[0][0].shape[1]
        nn_output_shape = train_dataset[0][1].shape[0]

    # Build model args and instantiate mSHRED on the selected device.
    args = make_model_args(
        encoder=encoder,
        nn_input_shape=nn_input_shape,
        nn_output_shape=nn_output_shape,
        LAGS=LAGS,
        TR_ENCODER_DEPTH=TR_ENCODER_DEPTH,
        TR_N_HEADS=TR_N_HEADS,
        TR_DIM_FEEDFORWARD=TR_DIM_FEEDFORWARD,
        TR_HIDDEN_SIZE=TR_HIDDEN_SIZE,
        TR_DROPOUT=TR_DROPOUT,
        LSTM_NUM_LAYERS=LSTM_NUM_LAYERS,
        GRUD_HIDDEN_SIZE=GRUD_HIDDEN_SIZE,
        GRUD_DROPOUT=GRUD_DROPOUT,
        X_mean=X_mean,
        double_channels=double_channels,
    )
    model = mSHRED(args=args).to(device)

    # Train model and capture train/validation MRE traces.
    train_errors, valid_errors = fit2(
        model,
        train_dataset,
        val_dataset,
        optimizer=OPTIMIZER,
        lr=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        verbose=VERBOSE_TRAIN,
        lr_scheduler=LR_SCHEDULER,
        scheduler_step_size=SCHEDULER_STEP_SIZE,
        masking_type=masking_type,
    )

    # Plot learning curves when visualization is enabled.
    if make_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(train_errors, label="Train MRE")
        plt.plot(valid_errors, label="Val MRE")
        plt.xlabel("Epoch")
        plt.ylabel("MRE")
        plt.title(f"{label}: Training and Validation MRE")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Evaluate trained model on train/validation/test splits.
    train_mse, train_mre = evaluate2(model, train_dataset, masking_type=masking_type)
    val_mse, val_mre = evaluate2(model, val_dataset, masking_type=masking_type)
    test_mse, test_mre = evaluate2(model, test_dataset, masking_type=masking_type)

    # Store metrics for cross-experiment comparison in notebook tables.
    ERRORS.append(
        {
            "model": f"(mSHRED): {encoder} + MLP",
            "type": exp_type,
            "train-mse": float(train_mse),
            "val-mse": float(val_mse),
            "test-mse": float(test_mse),
            "train-mre": float(train_mre),
            "val-mre": float(val_mre),
            "test-mre": float(test_mre),
        }
    )

    # Print quick summary metric for current run.
    print(f"[{label}] test MRE: {test_mre:.5f}")

    # Visual separator in notebook output.
    print("\n", "-" * 100, "\n")

    # Optionally generate qualitative reconstruction diagnostics.
    engine = None
    if make_plot:
        engine = plot_swe_reconstruction(
            manager=manager,
            test_dataset=test_dataset,
            model=model,
            label=label,
            H=H,
            Utheta=Utheta,
            Uphi=Uphi,
            Lx=Lx,
            Ly=Ly,
        )

    # Release large objects to reduce notebook memory pressure.
    del model, train_dataset, val_dataset, test_dataset
    if engine is not None:
        del engine

    # Return accumulated errors as a dataframe for immediate display/export.
    return pd.DataFrame(ERRORS)

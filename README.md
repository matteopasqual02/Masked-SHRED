# Masked-SHRED

**Real-time state reconstruction from sparse, asynchronous, and incomplete sensor measurements.**
 
Real-world sensing systems are inherently imperfect: sensors fail, disconnect intermittently, sample at asynchronous rates, and are subject to noise — producing measurement streams that are irregular, incomplete, and heterogeneous. Yet most sensing strategies for real-time state reconstruction assume full observability, a constraint that severely limits their applicability in operational settings.
 
**Masked-SHRED** addresses this gap by introducing and systematically evaluating a hierarchy of transformer-based masking mechanisms — channel-wise, hidden-state, and multi-head self-attention masking — within the [Shallow Recurrent Decoder (SHRED)](https://arxiv.org/abs/2301.12011) framework. By incorporating the binary missingness mask at multiple levels of the architecture, the model becomes explicitly aware of sensor availability, enabling high-dimensional state estimation from sparse, asynchronous, and missing sensor measurements.
 
We further introduce a **multi-step decoding training objective**: by decoding at intermediate timesteps and applying a weighted reconstruction loss, the model receives auxiliary supervision throughout the temporal axis, encouraging smoother and more coherent latent trajectories.

![Figure 1](Images_final/fig1.svg)
 
Across synthetic benchmarks and a real-world hydrodynamic dataset, we show that:
 
1. **Transformer-based masking outperforms alternatives.** It beats standard imputation techniques, recurrent architectures designed for incomplete time-series, and the conventional SHRED framework restricted to continuously available sensors — especially under heterogeneous missing-data patterns and over extended prediction horizons.
2. **Multi-step decoding improves robustness.** Auxiliary supervision during training enhances reconstruction fidelity and robustness to heterogeneous missing-data patterns.
3. **No added inference cost.** Both contributions preserve the computational efficiency of the original SHRED framework — masking is integrated directly into the encoder, and auxiliary decoding branches are discarded after training.

## Table of Contents
 
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

 
## Installation
 
```bash
git clone https://github.com/matteopasqual02/Masked-SHRED.git
cd masked-shred
uv sync --all-groups
```

## Quickstart
 
```python
# --- Standard libraries and dataset loading ---
# import numpy as np
# ...
 
# --- Masked-SHRED imports ---
from src.datamanager import DataManager
from src.engine import SHREDEngine
from src.train import fit, evaluate
from src.mSHRED.mSHRED import mSHRED
from src.utils import plot_training_curves, evaluate_model, plot_shred_reconstruction
 
# --- Load or build your data ---
# Expected shapes: sensor measurements (time, n_sensors) -> full field (time, space)
H, U, V = ...            # e.g. surface elevation, U/V velocity fields
measurements = ...        # sparse sensor measurement stream
 
# --- Data manager: builds lagged, masked train/val/test datasets ---
manager = DataManager(lags=LAGS, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE)
manager.add_data(H, id="Surface elevation", compress=COMPRESS_H, measurements=measurements)
manager.add_data(U, id="U velocity", compress=COMPRESS_U)
manager.add_data(V, id="V velocity", compress=COMPRESS_V)
 
train_dataset, val_dataset, test_dataset = manager.prepare(verbose=True, masking_type=MASKING_TYPE)
nn_input_shape = train_dataset.X.shape[-1]
nn_output_shape = train_dataset.Y.shape[-1]
 
# --- Model definition ---
model_args = dict(
    # Encoder
    encoder="transformer_learned_mask",
    encoder_depth=ENCODER_DEPTH,
    input_size=nn_input_shape,
    n_heads=N_HEADS,
    dim_feedforward=DIM_FEEDFORWARD,
    hidden_size=HIDDEN_SIZE,
    input_length=LAGS,
    double_channels=True,
    # Decoder
    decoder=DECODER,
    decoder_hidden_sizes=DECODER_HIDDEN_SIZES,
    output_size=nn_output_shape,
    dropout=DROPOUT,
)
mshred = mSHRED(args=model_args).to(device)
 
# --- Train ---
train_errors, valid_errors = fit(
    mshred, train_dataset, val_dataset,
    optimizer=OPTIMIZER, lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, patience=PATIENCE,
    verbose=VERBOSE_TRAIN, lr_scheduler=LR_SCHEDULER, scheduler_step_size=SCHEDULER_STEP_SIZE,
    masking_type=MASKING_TYPE,
)
 
# --- Evaluate ---
plot_training_curves(train_errors, valid_errors)
test_error = evaluate_model(
    evaluate, mshred, train_dataset, val_dataset, test_dataset,
    model=f"(mSHRED): TR+{DECODER}",
    type="(T.3.G): masking x2 channels + MHSA learned",
    masking_type=MASKING_TYPE,
)
 
# --- Reconstruct full state from sparse sensors ---
engine = SHREDEngine(manager, mshred)
test_reconstruction = engine.sensor_to_reconstruction_dataset(test_dataset)
plot_shred_reconstruction(test_reconstruction)
```
 
## Repository Structure

```text
src/
├── __init__.py
├── datamanager.py
├── engine.py
├── train.py
├── utils.py
└── mSHRED/
    ├── __init__.py
    ├── mSHRED.py
    ├── decoders/
    │   ├── CNN.py
    │   └── MLP.py
    └── encoders/
        ├── attnGRUD.py
        ├── GRUD.py
        ├── LSTM.py
        ├── transformer_attn_mask.py
        ├── transformer_hidden_mask.py
        ├── transformer_learned_mask.py
        └── transformer.py
```

## Results

Full experimental results, including synthetic benchmarks and the real-world hydrodynamic dataset, are available under [`Notebooks_results/`](Notebooks_results/).

The most important notebooks are listed here.

- DGF (simulated data)

  | Code | Notebook | Description |
  | --- | --- | --- |
  | 0 | [0-DGF.ipynb](Notebooks_results/DGF/0-DGF.ipynb) | Tutorial |
  | 1 | [1-DGF_results.ipynb](Notebooks_results/DGF/1-DGF_results.ipynb) | Results with synthetic sensor failures |
  | 2 | [2-DGF_multi_fidelity.ipynb](Notebooks_results/DGF/2-DGF_multi_fidelity.ipynb) | Results with non-overlapping sensors |

- HYDRO (real data)

  | Code | Notebook | Description |
  | --- | --- | --- |
  | 0 | [0-HYDRO_data_availability.ipynb](Notebooks_results/HYDRO/0-HYDRO_data_availability.ipynb) | Data availability exploration |
  | 0B | [0B-HYDRO_data_exploration.ipynb](Notebooks_results/HYDRO/0B-HYDRO_data_exploration.ipynb) | Dataset exploration |
  | 1 | [1-HYDRO_baseline.ipynb](Notebooks_results/HYDRO/1-HYDRO_baseline.ipynb) | Tutorial on baseline models |
  | 2 | [2-HYDRO_masking.ipynb](Notebooks_results/HYDRO/2-HYDRO_masking.ipynb) | Tutorial on masking models |
  | 3 | [3-HYDRO_less_sensors.ipynb](Notebooks_results/HYDRO/3-HYDRO_less_sensors.ipynb) | Tutorial on extra sparse sensing settings |
  | 4 | [4-HYDRO_results.ipynb](Notebooks_results/HYDRO/4-HYDRO_results.ipynb) | Final results |
  | 4B | [4B-HYDRO_results_ErrorAccumulation.ipynb](Notebooks_results/HYDRO/4B-HYDRO_results_ErrorAccumulation.ipynb) | Error accumulation: baseline vs. best-performing model |
  | 4C | [4C-HYDRO_paper_reconstruction_plot.ipynb](Notebooks_results/HYDRO/4C-HYDRO_paper_reconstruction_plot.ipynb) | Paper reconstruction plot |
  | 5 | [5-HYDRO_hyperparameter_tuning.ipynb](Notebooks_results/HYDRO/5-HYDRO_hyperparameter_tuning.ipynb) | Hyperparameter tuning |
  | 6 | [6-HYDRO_ROM_training.ipynb](Notebooks_results/HYDRO/6-HYDRO_ROM_training.ipynb) | ROM training |
  | 7 | [7-HYDRO_LAGGED_training.ipynb](Notebooks_results/HYDRO/7-HYDRO_LAGGED_training.ipynb) | Lagged training |
  | 7B | [7B-HYDRO_LAGGED_embedding_smoothness.ipynb](Notebooks_results/HYDRO/7B-HYDRO_LAGGED_embedding_smoothness.ipynb) | Embedding smoothness analysis |
  | 7C | [7C-HYDRO_LAGGED_new_plot_tuning.ipynb](Notebooks_results/HYDRO/7C-HYDRO_LAGGED_new_plot_tuning.ipynb) | Hyperparameter tuning analysis |
  | FINAL 1 | [FINAL-1-HYDRO_training.ipynb](Notebooks_results/HYDRO/FINAL-1-HYDRO_training.ipynb) | Paper result training |
  | FINAL 2 | [FINAL-2-HYDRO_training_lagged.ipynb](Notebooks_results/HYDRO/FINAL-2-HYDRO_training_lagged.ipynb) | Paper result lagged training |
  | z-MSE/MRE | [z-Table-MSE-MRE.ipynb](Notebooks_results/HYDRO/z-Table-MSE-MRE.ipynb) | Paper results table in MSE/MRE |
  | z-RMSE/NMAE | [z-Table-RMSE-NMAE.ipynb](Notebooks_results/HYDRO/z-Table-RMSE-NMAE.ipynb) | Paper results table in RMSE/NMAE |

- ORESUND (real data)

  | Code | Notebook | Description |
  | --- | --- | --- |
  | 0 | [0-ORESUND_data_availability.ipynb](Notebooks_results/ORESUND/0-ORESUND_data_availability.ipynb) | Data availability exploration |
  | 0B | [0B-ORESUND_data_exploration.ipynb](Notebooks_results/ORESUND/0B-ORESUND_data_exploration.ipynb) | Dataset exploration |
  | 1 | [1-ORESUND_RESULT.ipynb](Notebooks_results/ORESUND/1-ORESUND_RESULT.ipynb) | Results |
  | 1B | [1B-ORESUND_reconstruction_plot.ipynb](Notebooks_results/ORESUND/1B-ORESUND_reconstruction_plot.ipynb) | Paper reconstruction plot |
  | 2 | [2-ORESUND_LAGGED.ipynb](Notebooks_results/ORESUND/2-ORESUND_LAGGED.ipynb) | Lagged training |
  | FINAL 1 | [FINAL-1-ORESUND_training.ipynb](Notebooks_results/ORESUND/FINAL-1-ORESUND_training.ipynb) | Paper result training |
  | FINAL 2 | [FINAL-2-ORESUND_training_lagged.ipynb](Notebooks_results/ORESUND/FINAL-2-ORESUND_training_lagged.ipynb) | Paper result lagged training |
  | z-MSE/MRE | [z-Table-MSE-MRE.ipynb](Notebooks_results/ORESUND/z-Table-MSE-MRE.ipynb) | Paper results table in MSE/MRE |
  | z-RMSE/NMAE | [z-Table-RMSE-NMAE.ipynb](Notebooks_results/ORESUND/z-Table-RMSE-NMAE.ipynb) | Paper results table in RMSE/NMAE |


- SST (simulated data)
  - [0-SST_downloader.ipynb](Notebooks/SST/0-SST_downloader.ipynb) — dataset download
  - [1-SST_init.ipynb](Notebooks/SST/1-SST_init.ipynb) — tutorial
  - [2-SST_results.ipynb](Notebooks/SST/2-SST_results.ipynb) — results
- SST + BUOY DATA (real data)
  - [3-SST_BuoyData.ipynb](Notebooks/SST/3-SST_BuoyData.ipynb) — buoy-data tutorial
  - [4-SST_BuoyData_results.ipynb](Notebooks/SST/4-SST_BuoyData_results.ipynb) — results
  - [5-SST_BuoyData_cs_shred.ipynb](Notebooks/SST/5-SST_BuoyData_cs_shred.ipynb) — CS-SHRED results
- SWE (simulated data)
  - [0-SWE-ROM_exploration.ipynb](Notebooks/SWE/0-SWE-ROM_exploration.ipynb) — dataset download and exploration
  - [1-SWE-ROM.ipynb](Notebooks/SWE/1-SWE-ROM.ipynb) — tutorial
  - [2-SWE-ROM_results.ipynb](Notebooks/SWE/2-SWE-ROM_results.ipynb) — results 

## Citation

TODO

## License
 
This project is released under the [MIT License](LICENSE)

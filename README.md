# Masked-SHRED

**Fault-tolerant sparse sensing for real-time state reconstruction from sparse, asynchronous, and missing sensor measurements.**
 
Real-world sensing systems are inherently imperfect: sensors may fail, intermittently disconnect, operate with asynchronous sampling rates, and are subject to noise, leading to measurement streams that are irregular, incomplete, and heterogeneous. Nevertheless, sensing strategies for real-time state reconstruction typically assume synchronous and continuously active sensors, a constraint that substantially limits their applicability in operational environments.
 
In this work, **Masked-SHRED** introduces fault-tolerant sparse sensing strategies in the context of the [Shallow Recurrent Decoder (SHRED)](https://arxiv.org/abs/2301.12011) to enable high-dimensional state estimation from sparse, asynchronous, and missing sensor measurements. Specifically, we consider naive imputation strategies, fault-tolerant recurrent temporal encoders, and four different transformer-based masking mechanisms to make SHRED explicitly aware of sensor availability through the binary data-missingness mask.
 
Moreover, we further introduce a **multi-step decoding training objective** for SHRED, resulting in smoother latent trajectories and more robust predictions.

![Figure 1](Images_final/fig1.svg)
 
Throughout a synthetic benchmark and three real-world problems in hydrodynamic and ocean modeling, we show that:
 
1. Transformer-based masking mechanisms outperform standard imputation techniques and fault-tolerant recurrent architectures.
2. The multi-step decoding training objective enhances reconstruction fidelity and robustness in the presence of heterogeneous missing-data patterns.
3. Computational SHRED efficiency is preserved, with no additional inference-time overhead.

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
# --- Standard libraries ---
import numpy as np
import torch

# --- Masked-SHRED imports ---
from src.datamanager import DataManager
from src.engine import SHREDEngine
from src.train import fit, evaluate
from src.mSHRED.mSHRED import mSHRED
from src.utils import evaluate_model, plot_training_curves

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAGS = 20
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.8, 0.1, 0.1
MASKING_TYPE = "double_channels"
COMPRESS_H = COMPRESS_U = COMPRESS_V = 50
ENCODER_DEPTH = 2
N_HEADS = 4
DIM_FEEDFORWARD = 128
HIDDEN_SIZE = 64
DECODER = "mlp"
DECODER_HIDDEN_SIZES = [64, 128]
DROPOUT = 0.1
OPTIMIZER = "AdamW"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
PATIENCE = 20
VERBOSE_TRAIN = True
LR_SCHEDULER = None
SCHEDULER_STEP_SIZE = 10
 
# --- Load or build your data ---
# Replace these placeholders with arrays of shape (time, space) and
# measurements of shape (time, n_sensors).
H, U, V = load_your_fields()
measurements = load_your_sensor_measurements()
 
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

# --- Make a video ---
from src.utils import make_video
make_video(
    [ground_truth[-30:], test_reconstruction[-30:]],
    plot=plot_snapshot,
    titles=["Ground truth", "Best model"],
    save=True,
    name="last_month.gif",
    fps=10,
)
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

Full experimental results are available under [`Notebooks_results/`](Notebooks_results/). The notebooks are grouped by dataset:

| Dataset | Type | Main results |
| --- | --- | --- |
| DGF | Simulated | [DGF results](Notebooks_results/DGF/1-DGF_results.ipynb) |
| HYDRO | Real-world hydrodynamic data | [HYDRO results](Notebooks_results/HYDRO/4-HYDRO_results.ipynb) |
| ORESUND | Real-world data | [ORESUND results](Notebooks_results/ORESUND/1-ORESUND_RESULT.ipynb) |
| SST | Simulated and buoy measurements | [SST results](Notebooks_results/SST/2-SST_results_simulated_sensors.ipynb) |
| SWE | Simulated | [SWE results](Notebooks_results/SWE/1-SWE-ROM_results.ipynb) |

### DGF (simulated data)

| Code | Notebook | Description |
| --- | --- | --- |
| 0 | [0-DGF.ipynb](Notebooks_results/DGF/0-DGF.ipynb) | Tutorial |
| 1 | [1-DGF_results.ipynb](Notebooks_results/DGF/1-DGF_results.ipynb) | Results with synthetic sensor failures |
| 2 | [2-DGF_multi_fidelity.ipynb](Notebooks_results/DGF/2-DGF_multi_fidelity.ipynb) | Results with non-overlapping sensors and multi-fidelity data |

### HYDRO (real-world hydrodynamic data)

| Code | Notebook | Description |
| --- | --- | --- |
| 0 | [0-HYDRO_data_availability.ipynb](Notebooks_results/HYDRO/0-HYDRO_data_availability.ipynb) | Data availability exploration |
| 0B | [0B-HYDRO_data_exploration.ipynb](Notebooks_results/HYDRO/0B-HYDRO_data_exploration.ipynb) | Dataset exploration |
| 1 | [1-HYDRO_baseline.ipynb](Notebooks_results/HYDRO/1-HYDRO_baseline.ipynb) | Baseline models |
| 2 | [2-HYDRO_masking.ipynb](Notebooks_results/HYDRO/2-HYDRO_masking.ipynb) | Masking models |
| 3 | [3-HYDRO_less_sensors.ipynb](Notebooks_results/HYDRO/3-HYDRO_less_sensors.ipynb) | Extra-sparse sensing settings |
| 4 | [4-HYDRO_results.ipynb](Notebooks_results/HYDRO/4-HYDRO_results.ipynb) | Final results |
| 4B | [4B-HYDRO_results_ErrorAccumulation.ipynb](Notebooks_results/HYDRO/4B-HYDRO_results_ErrorAccumulation.ipynb) | Error accumulation analysis |
| 4C | [4C-HYDRO_paper_reconstruction_plot.ipynb](Notebooks_results/HYDRO/4C-HYDRO_paper_reconstruction_plot.ipynb) | Paper reconstruction plot |
| 5 | [5-HYDRO_hyperparameter_tuning.ipynb](Notebooks_results/HYDRO/5-HYDRO_hyperparameter_tuning.ipynb) | Hyperparameter tuning |
| 6 | [6-HYDRO_ROM_training.ipynb](Notebooks_results/HYDRO/6-HYDRO_ROM_training.ipynb) | ROM training |
| 7 | [7-HYDRO_LAGGED_training.ipynb](Notebooks_results/HYDRO/7-HYDRO_LAGGED_training.ipynb) | Lagged training |
| 7B | [7B-HYDRO_LAGGED_embedding_smoothness.ipynb](Notebooks_results/HYDRO/7B-HYDRO_LAGGED_embedding_smoothness.ipynb) | Embedding smoothness analysis |
| 7C | [7C-HYDRO_LAGGED_new_plot_tuning.ipynb](Notebooks_results/HYDRO/7C-HYDRO_LAGGED_new_plot_tuning.ipynb) | Hyperparameter tuning analysis |
| FINAL 1 | [FINAL-1-HYDRO_training.ipynb](Notebooks_results/HYDRO/FINAL-1-HYDRO_training.ipynb) | Paper result training |
| FINAL 2 | [FINAL-2-HYDRO_training_lagged.ipynb](Notebooks_results/HYDRO/FINAL-2-HYDRO_training_lagged.ipynb) | Paper result lagged training |
| MSE/MRE | [z-Table-MSE-MRE.ipynb](Notebooks_results/HYDRO/z-Table-MSE-MRE.ipynb) | Paper results table in MSE/MRE |
| RMSE/NMAE | [z-Table-RMSE-NMAE.ipynb](Notebooks_results/HYDRO/z-Table-RMSE-NMAE.ipynb) | Paper results table in RMSE/NMAE |

### ORESUND (real-world data)

| Code | Notebook | Description |
| --- | --- | --- |
| 0 | [0-ORESUND_data_availability.ipynb](Notebooks_results/ORESUND/0-ORESUND_data_availability.ipynb) | Data availability exploration |
| 0B | [0B-ORESUND_data_exploration.ipynb](Notebooks_results/ORESUND/0B-ORESUND_data_exploration.ipynb) | Dataset exploration |
| 1 | [1-ORESUND_RESULT.ipynb](Notebooks_results/ORESUND/1-ORESUND_RESULT.ipynb) | Final results |
| 1B | [1B-ORESUND_reconstruction_plot.ipynb](Notebooks_results/ORESUND/1B-ORESUND_reconstruction_plot.ipynb) | Paper reconstruction plot |
| 2 | [2-ORESUND_LAGGED.ipynb](Notebooks_results/ORESUND/2-ORESUND_LAGGED.ipynb) | Lagged training |
| FINAL 1 | [FINAL-1-ORESUND_training.ipynb](Notebooks_results/ORESUND/FINAL-1-ORESUND_training.ipynb) | Paper result training |
| FINAL 2 | [FINAL-2-ORESUND_training_lagged.ipynb](Notebooks_results/ORESUND/FINAL-2-ORESUND_training_lagged.ipynb) | Paper result lagged training |
| MSE/MRE | [z-Table-MSE-MRE.ipynb](Notebooks_results/ORESUND/z-Table-MSE-MRE.ipynb) | Paper results table in MSE/MRE |
| RMSE/NMAE | [z-Table-RMSE-NMAE.ipynb](Notebooks_results/ORESUND/z-Table-RMSE-NMAE.ipynb) | Paper results table in RMSE/NMAE |

### SST (simulated and buoy measurements)

| Code | Notebook | Description |
| --- | --- | --- |
| 0 | [0-SST_downloader.ipynb](Notebooks_results/SST/0-SST_downloader.ipynb) | Dataset download |
| 1 | [1-SST_tutorial.ipynb](Notebooks_results/SST/1-SST_tutorial.ipynb) | Tutorial |
| 2 | [2-SST_results_simulated_sensors.ipynb](Notebooks_results/SST/2-SST_results_simulated_sensors.ipynb) | Simulated-sensor results |
| 3 | [3-SST_BuoyData_exploration.ipynb](Notebooks_results/SST/3-SST_BuoyData_exploration.ipynb) | Buoy-data exploration |
| FINAL 1 | [FINAL-1-SST_train_T3G.ipynb](Notebooks_results/SST/FINAL-1-SST_train_T3G.ipynb) | Final T3G training |
| FINAL 2 | [FINAL-2-SST_train_CS.ipynb](Notebooks_results/SST/FINAL-2-SST_train_CS.ipynb) | Final CS-SHRED training |
| RMSE/NMAE | [z-Table-RMSE-NMAE.ipynb](Notebooks_results/SST/z-Table-RMSE-NMAE.ipynb) | Paper results table in RMSE/NMAE |

### SWE (simulated data)

| Code | Notebook | Description |
| --- | --- | --- |
| 0 | [0-SWE-ROM_data_exploration.ipynb](Notebooks_results/SWE/0-SWE-ROM_data_exploration.ipynb) | Dataset exploration |
| 1 | [1-SWE-ROM_results.ipynb](Notebooks_results/SWE/1-SWE-ROM_results.ipynb) | Final results |
| 1B | [1B-SWE-ROM_results_baseline.ipynb](Notebooks_results/SWE/1B-SWE-ROM_results_baseline.ipynb) | Baseline results |
| 1C | [1C-SWE-ROM_reconstruction_plot.ipynb](Notebooks_results/SWE/1C-SWE-ROM_reconstruction_plot.ipynb) | Paper reconstruction plot |
| FINAL 1 | [FINAL-1-SWE_training.ipynb](Notebooks_results/SWE/FINAL-1-SWE_training.ipynb) | Final training |
| RMSE/NMAE | [z-Table-RMSE-NMAE.ipynb](Notebooks_results/SWE/z-Table-RMSE-NMAE.ipynb) | Paper results table in RMSE/NMAE |


## Citation

If you use this code for your work, please cite
```bibtex
@misc{pasqual2026faulttolerantshallowrecurrentdecoders,
      title={Fault-tolerant shallow recurrent decoders for sparse sensing with missing sensor measurements}, 
      author={Matteo Romilio Pasqual and Matteo Tomasetto and J. Nathan Kutz and Andrea Manzoni},
      year={2026},
      eprint={2611.04893},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.04893}, 
}

## License
 
This project is released under the [MIT License](LICENSE)

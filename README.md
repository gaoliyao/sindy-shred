# SINDy-SHRED

**Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder Networks**

[![arXiv](https://img.shields.io/badge/arXiv-2501.13329-b31b1b.svg)](https://arxiv.org/abs/2501.13329)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?logo=youtube)](https://www.youtube.com/watch?v=UYDfWJxvKGw)

SINDy-SHRED learns low-dimensional latent dynamics from sparse sensor measurements and decodes those dynamics back into full spatiotemporal fields. It combines SHRED-style sequence encoding and reconstruction with SINDy-based sparse equation discovery, so the learned latent trajectories can be forecast with interpretable governing equations.

<img width="924" alt="SINDy-SHRED Architecture" src="https://github.com/user-attachments/assets/4b37563a-f6cc-49e5-8bd9-44842b70959f" />

## What It Does

- Encodes sparse sensor histories with GRU layers.
- Reconstructs high-dimensional fields with a shallow decoder network.
- Regularizes training with SINDy terms in the latent space.
- Discovers post-hoc symbolic latent dynamics with PySINDy.
- Supports first-order systems, `z' = f(z)`, and second-order systems, `z'' = f(z, z')`.
- Provides threshold search for sparse, stable SINDy models.

Example applications include sea surface temperature forecasting, flow over a cylinder, isotropic turbulence, pendulum video prediction, and synthetic PDE systems.

## Paper

The accompanying preprint is available on arXiv:

**Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks**  
https://arxiv.org/abs/2501.13329

## Installation

SINDy-SHRED requires Python 3.9 or newer. Install PyTorch first so that you get the correct build for your CPU or GPU, then install the package.

### 1. Install PyTorch

Choose the command that matches your hardware. If you are using an NVIDIA GPU, check your supported CUDA version with `nvidia-smi`.

```bash
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch
```

Verify the install:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 2. Install SINDy-SHRED

```bash
git clone https://github.com/gaoliyao/sindy-shred.git
cd sindy-shred
pip install -e ".[notebooks]"
```

For `uv` users:

```bash
git clone https://github.com/gaoliyao/sindy-shred.git
cd sindy-shred
uv venv .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[notebooks]"
```

For `conda` users:

```bash
git clone https://github.com/gaoliyao/sindy-shred.git
cd sindy-shred
conda create -n sindyshred python=3.11
conda activate sindyshred
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[notebooks]"
```

Confirm the package imports:

```bash
python -c "import torch, pysindy, sindy_shred; print('SINDy-SHRED ready')"
```

## Data

Place datasets in the `Data/` directory. Example data are available here:

https://drive.google.com/file/d/1IrKFsYEcUL8xxZ0PUSLC3VrpTvVneDhj/view?usp=sharing

The repository includes notebooks for the FitzHugh-Nagumo example, sea surface temperature data, and flow over a cylinder.

## Quick Start

Run a notebook locally:

```bash
jupyter lab notebooks/fitzhugh_nagumo.ipynb
```

Or open the hosted Colab notebook:

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing)

## Python API

The high-level `SINDySHRED` class handles scaling, sensor-window construction, model training, latent equation discovery, and decoding.

```python
from sindy_shred import SINDySHRED

model = SINDySHRED(
    latent_dim=3,
    poly_order=2,
    hidden_layers=2,
    l1=350,
    l2=400,
    dropout=0.1,
    batch_size=128,
    num_epochs=200,
    lr=1e-3,
    threshold=0.05,
    sindy_regularization=10.0,
)

model.fit(
    num_sensors=3,
    dt=1 / 52,
    x_to_fit=data,                  # shape: (time, space)
    lags=52,
    train_length=1000,
    validate_length=30,
    sensor_locations=sensor_locations,
)

model.sindy_identify(threshold=0.05, plot_result=True)

z_pred = model.sindy_predict(split="train", init_from="last")
forecast_scaled = model.shred_decode(z_pred)
forecast = model.forecast(n_steps=100, return_scaled=False)
```

Useful methods:

| Method | Purpose |
| --- | --- |
| `fit(...)` | Train the SHRED encoder-decoder with SINDy regularization. |
| `sindy_identify(...)` | Discover sparse latent dynamics from trained GRU trajectories. |
| `auto_tune_threshold(...)` | Search for a sparse, stable SINDy threshold. |
| `sindy_predict(...)` | Simulate the discovered latent dynamics. |
| `shred_decode(...)` | Decode latent trajectories into scaled physical states. |
| `forecast(...)` | Predict latent dynamics, decode them, and optionally invert scaling. |
| `sensor_recon(...)` | Reconstruct full states directly from sparse sensor inputs. |

## Lower-Level Access

For custom training loops or experiments, use the underlying network and utilities directly:

```python
from sindy_shred import sindy
from sindy_shred.sindy_shred_net import SINDy_SHRED_net, fit

library_dim = sindy.library_size(
    n=latent_dim,
    poly_order=poly_order,
    use_sine=False,
    include_constant=True,
)

net = SINDy_SHRED_net(
    input_size=num_sensors,
    output_size=state_dim,
    hidden_size=latent_dim,
    hidden_layers=2,
    l1=350,
    l2=400,
    dropout=0.1,
    library_dim=library_dim,
    poly_order=poly_order,
    include_sine=False,
    dt=dt,
).to(device)

validation_errors = fit(
    net,
    train_dataset,
    valid_dataset,
    batch_size=128,
    num_epochs=600,
    lr=1e-3,
    threshold=0.25,
    sindy_regularization=10.0,
)
```

## Notebooks

| Notebook | Description |
| --- | --- |
| `notebooks/fitzhugh_nagumo.ipynb` | End-to-end synthetic FitzHugh-Nagumo workflow. |
| `notebooks/sea_surface_temperature.ipynb` | Sea surface temperature reconstruction and forecasting. |
| `notebooks/flow_over_cylinder.ipynb` | Flow-over-cylinder reconstruction and latent forecasting. |

Notebook outputs are written under `notebooks/results/`.

## Package Layout

| Module | Description |
| --- | --- |
| `sindy_shred.sindy_shred` | High-level `SINDySHRED` workflow class. |
| `sindy_shred.sindy_shred_net` | Core PyTorch network and training routine. |
| `sindy_shred.sindy` | SINDy library construction and sparse dynamics helpers. |
| `sindy_shred.miosr_utils` | MIOSR-based sparse model utilities. |
| `sindy_shred.processdata` | Data loading and preprocessing helpers. |
| `sindy_shred.plotting` | Plotting utilities for latent dynamics and reconstructions. |
| `sindy_shred.utils` | Device selection and dataset wrappers. |

## Results

### Sea Surface Temperature

<img width="1284" alt="SST Results" src="https://github.com/user-attachments/assets/ecb0bedc-ab58-46f5-81d5-51a2cc31e575" />

### Flow Over a Cylinder

<img width="1284" alt="Cylinder Flow" src="https://github.com/user-attachments/assets/f719b85e-c0cf-4f95-a195-fb331e3819b2" />

### Isotropic Turbulent Flow

<img width="1284" alt="Turbulent Flow" src="https://github.com/user-attachments/assets/c745884b-99c0-40e7-821a-86f4ebd0d34a" />

### Pendulum Video Prediction

<img width="1284" alt="Pendulum Prediction" src="https://github.com/user-attachments/assets/65e4260d-2723-4de1-ae9c-d95212cad939" />

### Loss Landscape

<img width="1284" alt="Loss Landscape" src="https://github.com/user-attachments/assets/0d81d984-4f0a-4144-b87b-c94b7d969a67" />

## Citation

If you use SINDy-SHRED in your research, please cite:

```bibtex
@misc{gao2025sparse,
  title={Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks},
  author={Mars Liyao Gao and Jan P. Williams and J. Nathan Kutz},
  year={2025},
  eprint={2501.13329},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2501.13329}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

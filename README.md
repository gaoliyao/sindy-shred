# SINDy-SHRED

**Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder Networks**

[![arXiv](https://img.shields.io/badge/arXiv-2501.13329-b31b1b.svg)](https://arxiv.org/abs/2501.13329)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?logo=youtube)](https://www.youtube.com/watch?v=UYDfWJxvKGw)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-sindy--shred-blue.svg)](https://github.com/gaoliyao/sindy-shred)

## Overview

<img width="924" alt="SINDy-SHRED Architecture" src="https://github.com/user-attachments/assets/4b37563a-f6cc-49e5-8bd9-44842b70959f" />

SINDy-SHRED combines **sparse dynamics identification** with **shallow recurrent decoder networks** to reconstruct full spatiotemporal fields from sparse sensor measurements while discovering **interpretable governing equations** in the latent space.

**Key Features:**

- **SINDy regularization** enforces sparse, interpretable dynamics in the learned latent space
- **Gated Recurrent Units (GRUs)** encode sequential sensor measurements into latent trajectories
- **Shallow decoder network (SDN)** reconstructs high-dimensional fields from latent variables
- **Post-hoc SINDy discovery** extracts symbolic governing equations from learned latent dynamics
- **Automatic threshold tuning** via nonparametric coefficient-based search
- **Support for 1st and 2nd order ODEs** (z' = f(z) and z'' = f(z, z'))

**Applications:**

- Sea Surface Temperature (SST) prediction
- Flow over a cylinder
- Isotropic turbulent flow
- Video forecasting (e.g., pendulum dynamics)
- Synthetic PDE systems

## Paper

**Title:** *Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder Networks (SINDy-SHRED)*

**Preprint:** [arXiv:2501.13329](https://arxiv.org/pdf/2501.13329)

---

## Installation

### Dependencies

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- PySINDy
- seaborn

### Dataset

Download the dataset and place it in the `Data/` directory:

[Google Drive: Dataset](https://drive.google.com/file/d/1IrKFsYEcUL8xxZ0PUSLC3VrpTvVneDhj/view?usp=sharing)

---

## Quick Start

A ready-to-run notebook is available on Google Colab:

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing)

---

## Usage

### High-Level API (Recommended)

The `SINDySHRED` class provides an end-to-end interface:

```python
from sindy_shred import SINDySHRED

# Initialize the model
model = SINDySHRED(
    latent_dim=3,
    poly_order=1,
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

# Fit to data
model.fit(
    num_sensors=3,
    dt=1/52.0,
    x_to_fit=data,  # shape: (time, space)
    lags=52,
    train_length=1000,
    validate_length=30,
    sensor_locations=sensor_locations,
)

# Discover governing equations
model.sindy_identify(threshold=0.05, plot_result=True)

# Predict latent dynamics and decode to physical space
z_predict = model.sindy_predict()
forecast = model.shred_decode(z_predict)

# Or use the convenience method
forecast = model.forecast(n_steps=100)

# Automatic threshold tuning
best_threshold, results = model.auto_tune_threshold(adaptive=True)
```

### Low-Level API

For finer control over training and inference:

```python
import torch
import pysindy as ps
from sindy_shred_net import SINDy_SHRED_net, fit
import sindy

# Calculate library dimension
library_dim = sindy.library_size(latent_dim, poly_order, include_sine=False, include_constant=True)

# Initialize the network
shred = SINDy_SHRED_net(
    input_size=num_sensors,
    output_size=state_dim,
    hidden_size=latent_dim,
    hidden_layers=2,
    l1=350,
    l2=400,
    dropout=0.1,
    library_dim=library_dim,
    poly_order=3,
    include_sine=False,
    dt=dt,
).to(device)

# Train with custom datasets
validation_errors = fit(
    shred,
    train_dataset,
    valid_dataset,
    batch_size=128,
    num_epochs=600,
    lr=1e-3,
    verbose=True,
    threshold=0.25,
    patience=5,
    sindy_regularization=10.0,
    thres_epoch=100,
)

# Extract latent trajectories
gru_outs, _ = shred.gru_outputs(train_dataset.X, sindy=True)
latent = gru_outs[:, 0, :].detach().cpu().numpy()

# Post-hoc SINDy discovery
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)
model.fit(latent_normalized, t=dt)
model.print()

# Simulate and decode predictions
z_sim = model.simulate(init_cond, t_array)
z_tensor = torch.tensor(z_denormalized, dtype=torch.float32).to(device)
physical_pred = shred.decode(z_tensor)  # Decode latent to physical space
```

---

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| `sst_sindy_shred_refactor.ipynb` | Sea Surface Temperature with high-level API |
| `sst_sindy_shred.ipynb` | Sea Surface Temperature with low-level API |
| `synthetic_data_sindy_shred_refactor.ipynb` | FitzHugh-Nagumo synthetic data with high-level API |
| `synthetic_data_sindy_shred.ipynb` | FitzHugh-Nagumo synthetic data with low-level API |
| `complex_data_sindy_shred_refactor.ipynb` | Complex dynamical systems with high-level API |
| `complex_data_sindy_shred.ipynb` | Complex dynamical systems with low-level API |

---

## Module Structure

| Module | Description |
|--------|-------------|
| `sindy_shred.py` | High-level `SINDySHRED` class for end-to-end workflows |
| `sindy_shred_net.py` | Core `SINDy_SHRED_net` neural network and training functions |
| `sindy.py` | SINDy library functions for sparse dynamics identification |
| `plotting.py` | Visualization utilities for latent space and predictions |
| `processdata.py` | Data loading and preprocessing utilities |
| `utils.py` | Helper functions (device selection, datasets) |

---

## Results

SINDy-SHRED achieves state-of-the-art performance compared to existing methods including Convolutional LSTM, PredRNN, ResNet, and SimVP.

### Sea Surface Temperature Prediction

<img width="1284" alt="SST Results" src="https://github.com/user-attachments/assets/ecb0bedc-ab58-46f5-81d5-51a2cc31e575" />

### Flow over a Cylinder

<img width="1284" alt="Cylinder Flow" src="https://github.com/user-attachments/assets/f719b85e-c0cf-4f95-a195-fb331e3819b2" />

### Isotropic Turbulent Flow

<img width="1284" alt="Turbulent Flow" src="https://github.com/user-attachments/assets/c745884b-99c0-40e7-821a-86f4ebd0d34a" />

### Pendulum Video Prediction

<img width="1284" alt="Pendulum Prediction" src="https://github.com/user-attachments/assets/65e4260d-2723-4de1-ae9c-d95212cad939" />

### Loss Landscape Visualization

<img width="1284" alt="Loss Landscape" src="https://github.com/user-attachments/assets/0d81d984-4f0a-4144-b87b-c94b7d969a67" />

---

## Citation

If you find SINDy-SHRED useful in your research, please cite:

```bibtex
@misc{gao2025sparse,
      title={Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks},
      author={Mars Liyao Gao and Jan P. Williams and J. Nathan Kutz},
      year={2025},
      eprint={2501.13329},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.13329},
}
```

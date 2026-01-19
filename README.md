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

SINDy-SHRED is a method for **spatiotemporal modeling** that integrates **sensing and model identification** using a **shallow recurrent decoder network**. It reconstructs full spatiotemporal fields from sparse sensor measurements while learning **interpretable latent space dynamics**.

**Key Features:**

- **SINDy regularization** enforces latent space dynamics that follow a sparse, interpretable functional form
- **Gated Recurrent Units (GRUs)** process sequential sensor measurements
- **Shallow decoder network** reconstructs high-dimensional fields from latent variables
- **Koopman-SHRED variant** enforces linear latent dynamics using Koopman theory
- **Minimal hyperparameter tuning** required; runs efficiently on standard hardware

**Applications:**

- Synthetic PDE data modeling
- Sea Surface Temperature (SST) prediction
- Long-term video forecasting

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

The `SINDyShred` class provides an end-to-end interface for data preprocessing, training, and SINDy discovery:

```python
from sindy_shred import SINDyShred

# Initialize the model
model = SINDyShred(
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

# Fit to data (handles preprocessing automatically)
model.fit(
    num_sensors=num_sensors,
    dt=1/52.0,
    x_to_fit=load_X,
    lags=52,
    train_length=1000,
    validate_length=30,
    sensor_locations=sensor_locations,
)

# Discover governing equations
model.sindy_identify(threshold=0.05, plot_result=True)

# Generate predictions
x_predict = model.sindy_predict()
output = model.shred_decode(x_predict)
```

### Low-Level API

For finer control over the training process:

```python
from sindy_shred_net import SINDy_SHRED_net, fit
import sindy

library_dim = sindy.library_size(latent_dim, poly_order, include_sine, True)

# Initialize the network
shred = SINDy_SHRED_net(
    num_sensors, m, hidden_size=3, hidden_layers=2, l1=350, l2=400, dropout=0.1,
    library_dim=library_dim, poly_order=3, include_sine=False, dt=1/52.0
)

# Train with custom datasets
validation_errors = fit(
    shred, train_dataset, valid_dataset, batch_size=128, num_epochs=600, lr=1e-3,
    verbose=True, threshold=0.25, patience=5, sindy_regularization=10.0, thres_epoch=100
)
```

---

## Module Structure

| Module | Description |
|--------|-------------|
| `sindy_shred.py` | High-level `SINDyShred` driver class for end-to-end workflows |
| `sindy_shred_net.py` | Core `SINDy_SHRED_net` neural network and training functions |
| `sindy.py` | SINDy library functions for sparse dynamics identification |
| `plotting.py` | Visualization utilities for latent space and reconstructions |
| `processdata.py` | Data loading and preprocessing utilities |

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

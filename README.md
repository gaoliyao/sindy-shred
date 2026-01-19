# SINDy-SHRED: Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder Networks

[![arXiv](https://img.shields.io/badge/arXiv-2501.13329-b31b1b.svg)](https://arxiv.org/abs/2501.13329)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?logo=youtube)](https://www.youtube.com/watch?v=UYDfWJxvKGw)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-sindy--shred-blue.svg)](https://github.com/gaoliyao/sindy-shred)



## 📌 Overview

<img width="924" alt="Screen Shot 2025-01-24 at 6 09 27 AM" src="https://github.com/user-attachments/assets/4b37563a-f6cc-49e5-8bd9-44842b70959f" />

SINDy-SHRED is a method for **spatiotemporal modeling of real-world data** that integrates **sensing and model identification** using a **shallow recurrent decoder network**. It efficiently reconstructs the full spatiotemporal field from a few sensor measurements while enforcing **interpretable latent space dynamics**.

- **Sparse Identification of Nonlinear Dynamics (SINDy)** regularization ensures latent space dynamics follow a **SINDy-class functional**.
- **Gated Recurrent Units (GRUs)** handle sequential sensor measurements.
- **Shallow decoder network** reconstructs the high-dimensional field from latent variables.
- **Koopman-SHRED** variant enforces **linear latent space dynamics** using Koopman theory.
- **Minimal hyperparameter tuning** required; runs efficiently on **laptop-level computing**.

SINDy-SHRED achieves **state-of-the-art performance** in various applications, including:
- ✅ **Synthetic PDE data** modeling
- ✅ **Sea Surface Temperature (SST) prediction** from real sensor data
- ✅ **Long-term video forecasting**

## 📝 Paper
📄 **Title:** *Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder networks (SINDy-SHRED)*  
🔗 **Preprint:** [arXiv:2501.13329](https://arxiv.org/pdf/2501.13329)  


---

## 🔧 Installation
### Dependencies
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- scikit-learn

## 🚀 Quick Start in Google Colab

We strongly encourage you to explore **SINDy-SHRED** by directly running our [Colab notebook](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing). The colab notebook allows you to easily reproduce experiments, and understand the workflow without any setup:

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing)

✨ **Just plug and play!** 🎸

### 📂 Dataset
Please download the dataset and place it into the `Data/` folder, as GitHub might mistakenly ignore large files.

**Download Link:**  🔗 [Google Drive: Dataset](https://drive.google.com/file/d/1IrKFsYEcUL8xxZ0PUSLC3VrpTvVneDhj/view?usp=sharing)


---

## 🚀 Usage

### 1️⃣ Train a SINDy-SHRED model (Recommended)

The `SINDyShred` class provides a high-level interface that handles data preprocessing, training, and SINDy discovery:

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

# Make predictions
x_predict = model.sindy_predict()
output = model.shred_decode(x_predict)
```

### 2️⃣ Low-level API (Advanced)

For more control, use the network class directly:

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

## 📁 Module Structure

| Module | Description |
|--------|-------------|
| `sindy_shred.py` | High-level `SINDyShred` driver class for end-to-end workflows |
| `sindy_shred_net.py` | Core `SINDy_SHRED_net` neural network and training functions |
| `sindy.py` | SINDy library functions for sparse dynamics identification |
| `plotting.py` | Visualization utilities for latent space, reconstructions, and predictions |
| `processdata.py` | Data loading and preprocessing utilities |

---

## 📊 Results & Benchmarks
SINDy-SHRED **outperforms state-of-the-art models** on real and synthetic datasets, including:
- ✅ **Convolutional LSTM**  
- ✅ **PredRNN**  
- ✅ **ResNet**  
- ✅ **SimVP**  

### 🌍 Real-World SST Data Example
<img width="1284" alt="Screen Shot 2025-01-24 at 6 11 38 AM" src="https://github.com/user-attachments/assets/ecb0bedc-ab58-46f5-81d5-51a2cc31e575" />

### 💨 Flow over a Cylinder Example
<img width="1284" alt="Screenshot 2025-04-01 at 2 52 27 AM" src="https://github.com/user-attachments/assets/f719b85e-c0cf-4f95-a195-fb331e3819b2" />

### 🌪 Isotropic Turbulent Flow Example
<img width="1284" alt="Screenshot 2025-04-01 at 2 57 15 AM" src="https://github.com/user-attachments/assets/c745884b-99c0-40e7-821a-86f4ebd0d34a" />

### 📽 Pendulum Video Prediction Example
<img width="1284" alt="Screen Shot 2025-01-24 at 6 12 36 AM" src="https://github.com/user-attachments/assets/65e4260d-2723-4de1-ae9c-d95212cad939" />

### 📉 Visualizing the Loss Landscape
<img width="1284" alt="Screenshot 2025-04-01 at 2 57 59 AM" src="https://github.com/user-attachments/assets/0d81d984-4f0a-4144-b87b-c94b7d969a67" />



---

## 📖 Citation
If you find **SINDy-SHRED** useful in your research, please cite:
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


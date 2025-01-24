# SINDy-SHRED: Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder Networks

[![arXiv](https://img.shields.io/badge/arXiv-2501.13329-b31b1b.svg)](https://arxiv.org/abs/2501.13329)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  

## 📌 Overview
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
```bash
git clone https://github.com/your-username/SINDy-SHRED.git
cd SINDy-SHRED
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- scikit-learn  

---

## 🚀 Usage
### 1️⃣ Train a SINDy-SHRED model
```python
import sindy_shred

latent_dim = 3
poly_order = 3
include_sine = False
library_dim = sindy_shred.library_size(latent_dim, poly_order, include_sine, True)

# Initialize and train the SINDy-SHRED model
shred = sindy_shred.SINDy_SHRED(
    num_sensors, m, hidden_size=latent_dim, hidden_layers=2, l1=350, l2=400, dropout=0.1, 
    library_dim=library_dim, poly_order=poly_order, include_sine=include_sine, dt=1/52.0*0.1, layer_norm=False
).to(device)

validation_errors = sindy_shred.fit(
    shred, train_dataset, valid_dataset, batch_size=128, num_epochs=600, lr=1e-3, verbose=True, 
    threshold=0.25, patience=5, sindy_regularization=10.0, optimizer="Lion", thres_epoch=100
)
```

---

## 📖 Citation
If you find **SINDy-SHRED** useful in your research, please cite:
```bibtex
@article{yourcitation2024,
  title={Sparse identification of nonlinear dynamics with Shallow Recurrent Decoder Networks},
  author={Gao, Liyao Mars and Williams, Jan P. and Kutz, J. Nathan},
  journal={arXiv preprint arXiv:2501.13329},
  year={2025}
}
```


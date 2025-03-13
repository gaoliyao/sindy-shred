# SINDy-SHRED: Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder Networks

[![arXiv](https://img.shields.io/badge/arXiv-2501.13329-b31b1b.svg)](https://arxiv.org/abs/2501.13329)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?logo=youtube)](https://www.youtube.com/watch?v=UYDfWJxvKGw)
[![Open in Colab](https://img.shields.io/badge/Colab-Open-orange?logo=googlecolab)]([https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing](https://colab.research.google.com/drive/1Xxw3P_x9a8iKZ6RPe2ZfTb8rJoWtPwTK?usp=sharing))


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

✨ **Just plug and play!**

### 📂 Dataset
Please download the dataset and place it into the `Data/` folder, as GitHub might mistakenly ignore large files.

**Download Link:**  🔗 [Google Drive: Dataset](https://drive.google.com/file/d/1IrKFsYEcUL8xxZ0PUSLC3VrpTvVneDhj/view?usp=sharing)


---

## 🚀 Usage
### 1️⃣ Define Train a SINDy-SHRED model
```python
import sindy_shred

library_dim = sindy_shred.library_size(latent_dim, poly_order, include_sine, True)

# Initialize and train the SINDy-SHRED model
shred = sindy_shred.SINDy_SHRED(
    num_sensors, m, hidden_size=3, hidden_layers=2, l1=350, l2=400, dropout=0.1, 
    library_dim=library_dim, poly_order=3, include_sine=False, dt=1/52.0*0.1, layer_norm=False
)

validation_errors = sindy_shred.fit(
    shred, train_dataset, valid_dataset, batch_size=128, num_epochs=600, lr=1e-3, verbose=True, 
    threshold=0.25, patience=5, sindy_regularization=10.0, optimizer="Lion", thres_epoch=100
)
```

---

## 📊 Results & Benchmarks
SINDy-SHRED **outperforms state-of-the-art models** on real and synthetic datasets, including:
- ✅ **Convolutional LSTM**  
- ✅ **PredRNN**  
- ✅ **ResNet**  
- ✅ **SimVP**  

### 🌍 Real-World SST Data Example
<img width="1284" alt="Screen Shot 2025-01-24 at 6 11 38 AM" src="https://github.com/user-attachments/assets/ecb0bedc-ab58-46f5-81d5-51a2cc31e575" />

### 📽 Pendulum Video Prediction Example
<img width="871" alt="Screen Shot 2025-01-24 at 6 12 36 AM" src="https://github.com/user-attachments/assets/65e4260d-2723-4de1-ae9c-d95212cad939" />


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


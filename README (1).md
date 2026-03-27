# 🔧 Predictive Maintenance — Industrial Fault Detection

> End-to-end predictive maintenance system: sensor simulation, anomaly detection (Autoencoder + LSTM + Transformers), RUL estimation, and interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-4CAF50)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🇬🇧 English

### Overview

This project implements a full predictive maintenance pipeline for industrial production lines. Starting from simulated multi-variate sensor data, it trains deep learning models to detect anomalies early, estimate **Remaining Useful Life (RUL)**, and visualize upcoming failures in real time.

**Core idea:** detect degradation patterns before failures occur — shifting maintenance from reactive to predictive.

### Key Features

- 🏭 **Realistic simulation** — 10,000+ sensor readings over 30 days via SimPy
- 🤖 **Hybrid deep learning** — Autoencoder for reconstruction error + LSTM + Transformer for temporal patterns
- ⏳ **RUL estimation** — Regression head predicting remaining cycles before failure
- 📊 **Interactive dashboard** — Real-time fault visualization via Streamlit
- 🔁 **MLOps-ready** — Modular code, versioned models, CI/CD compatible

### Results

| Model | F1-Score | Precision | Recall | RUL MAE |
|-------|----------|-----------|--------|---------|
| Autoencoder (baseline) | 0.81 | 0.84 | 0.78 | — |
| LSTM | 0.86 | 0.88 | 0.84 | 4.2 cycles |
| Autoencoder + LSTM + Transformer | **0.91** | **0.92** | **0.90** | **2.8 cycles** |

---

## 🇫🇷 Français

### Vue d'ensemble

Ce projet implémente un pipeline complet de maintenance prédictive pour des lignes de production industrielles. À partir de données capteurs multivariées simulées, des modèles de deep learning sont entraînés pour détecter les anomalies en amont, estimer la **Durée de Vie Résiduelle (RUL)**, et visualiser les pannes anticipées en temps réel.

**Idée centrale :** détecter les patterns de dégradation avant la panne — passer d'une maintenance réactive à une maintenance prescriptive.

### Fonctionnalités clés

- 🏭 **Simulation réaliste** — 10 000+ points capteurs sur 30 jours via SimPy
- 🤖 **Deep learning hybride** — Autoencoder (erreur de reconstruction) + LSTM + Transformer (patterns temporels)
- ⏳ **Estimation du RUL** — Tête de régression prédisant les cycles restants avant panne
- 📊 **Dashboard interactif** — Visualisation temps réel des pannes via Streamlit
- 🔁 **MLOps-ready** — Code modulaire, modèles versionnés, compatible CI/CD

---

## 🗂️ Repository Structure

```
predictive-maintenance/
│
├── data/
│   ├── raw/                        # Raw simulated sensor data
│   └── processed/                  # Normalized sequences, train/val/test splits
│
├── notebooks/
│   ├── 01_simulation_eda.ipynb     # SimPy simulation + EDA
│   ├── 02_preprocessing.ipynb      # Normalization, windowing, RUL labeling
│   ├── 03_autoencoder.ipynb        # Autoencoder training + threshold tuning
│   ├── 04_lstm_transformer.ipynb   # LSTM + Transformer training
│   └── 05_rul_estimation.ipynb     # RUL regression + evaluation
│
├── src/
│   ├── simulate.py                 # SimPy production line simulator
│   ├── preprocess.py               # Normalization, windowing, dataset splits
│   ├── models/
│   │   ├── autoencoder.py          # Convolutional Autoencoder (PyTorch)
│   │   ├── lstm_transformer.py     # LSTM + Transformer encoder (PyTorch)
│   │   └── rul_head.py             # RUL regression head
│   ├── train.py                    # Training loop + early stopping + checkpoints
│   ├── evaluate.py                 # Metrics, confusion matrix, RUL MAE
│   └── dashboard.py                # Streamlit real-time dashboard
│
├── models/                         # Saved model checkpoints (.pt)
├── outputs/                        # Figures, metrics, alert logs
├── .github/workflows/ci.yml        # CI pipeline (lint + test)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Methodology

### 1. Simulation (SimPy)
- Models a multi-machine production line with stochastic degradation
- 8 sensor channels per machine: temperature, vibration, pressure, rpm, current, torque, oil level, acoustics
- Fault injection at random intervals with configurable MTBF (Mean Time Between Failures)
- Outputs 10,000+ timestamped rows over 30 simulated days

### 2. Preprocessing
- MinMax normalization per channel
- Sliding window segmentation (window=50, stride=1)
- RUL label construction: linear decay from last known healthy state to failure
- Train / Validation / Test split: 70 / 15 / 15 (no data leakage across machines)

### 3. Anomaly Detection Models

**Autoencoder**
- Conv1D encoder → latent bottleneck → Conv1D decoder
- Trained on healthy sequences only
- Anomaly score = reconstruction error (MSE)
- Threshold tuned on validation set (95th percentile of healthy errors)

**LSTM + Transformer**
- LSTM captures short-term temporal dependencies
- Transformer encoder (multi-head attention) captures long-range patterns
- Binary classification head: normal vs anomalous

**Fusion**
- Ensemble: weighted average of Autoencoder score + LSTM/Transformer logit
- Calibrated probability output

### 4. RUL Estimation
- Regression head on top of LSTM/Transformer encoder
- Target: cycles remaining until failure
- Loss: Huber loss (robust to outliers)
- Evaluated with MAE and a custom asymmetric penalty (late predictions penalized more)

### 5. Dashboard (Streamlit)
- Live sensor feed simulation
- Real-time anomaly score gauge per machine
- RUL countdown per machine
- Alert log with severity levels (warning / critical)
- Historical fault timeline

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/HAMZAZAROUALI/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
```

### Run the full pipeline
```bash
# 1. Simulate sensor data
python src/simulate.py

# 2. Preprocess
python src/preprocess.py

# 3. Train models
python src/train.py --model autoencoder
python src/train.py --model lstm_transformer

# 4. Evaluate
python src/evaluate.py

# 5. Launch dashboard
streamlit run src/dashboard.py
```

---

## 📦 Requirements

```
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
simpy>=4.0
streamlit>=1.28
matplotlib>=3.7
seaborn>=0.12
plotly>=5.0
tqdm>=4.65
jupyter
```

---

## 👤 Author

**Hamza Zarouali** — AI & Data Science Engineer
[LinkedIn](https://linkedin.com/in/HAMZAZAROUALI) · [Email](mailto:hamzazarouali100@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

# Microsoft Stock Market Time-Series Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Data_Scaling-F7931E)

## 1. Introduction

This project is a comparative deep learning study focused on financial time-series forecasting. Using historical daily trading data for Microsoft stock (2001–2021), this repository implements, trains, and evaluates a highly modular Neural Network architecture to predict future closing prices.

Rather than deploying a single algorithm, this project evaluates the theoretical strengths, gradient dynamics, and computational efficiencies of various Recurrent Neural Network (RNN) engines, including Vanilla RNNs, LSTMs, and GRUs. It also explores advanced architectural concepts such as Deep Stacking and Residual (Skip) Connections to mitigate gradient degradation in deep temporal networks.

---

## 2. Table of Contents

1. [Introduction](#1-introduction)
2. [Project Folder Structure](#3-project-folder-structure)
3. [Methodology & Models Evaluated](#4-methodology--models-evaluated)
4. [Installation & Setup](#5-installation--setup)
5. [Usage & Configuration](#6-usage--configuration)
6. [Results & Output](#7-results--output)

---

## 3. Project Folder Structure

The codebase follows a strict, production-ready directory structure, decoupling the mathematical logic from the experimental configurations and data.

```text
📦 rnn_forecasting
 ┣ 📂 config
 ┃ ┣ 📜 config.json           # Global hyperparameters (batch size, epochs, etc.)
 ┃ ┗ 📜 experiments.json      # Roster of neural network architectures to test
 ┣ 📂 data
 ┃ ┗ 📜 dataset.csv           # Raw historical Microsoft stock data (2001-2021)
 ┣ 📂 results
 ┃ ┣ 📜 summary_[TIMESTAMP].json # Automatically generated experiment metrics & metadata
 ┃ ┗ 📜 stock_predictions_[TIMESTAMP].png # High-resolution export of model predictions
 ┣ 📂 src
 ┃ ┗ 📜 main.ipynb   # The core Jupyter Notebook containing the data pipeline and PyTorch logic
 ┣ 📜 requirements.txt        # Python dependencies
 ┗ 📜 README.md               # Project documentation
```

---

## 4. Methodology & Models Evaluated

To strictly prevent temporal data leakage, the dataset is split chronologically (70% Train, 15% Validation, 15% Test) and normalization (`MinMaxScaler`) is mathematically bounded exclusively to the training set.

The automated pipeline tests the following configurations:

- **Vanilla RNN (Baseline):** Demonstrates the vanishing gradient problem over a 20-day lookback window.
- **Shallow LSTM:** The industry standard for temporal gating and memory management.
- **Deep LSTM (3-Layer):** Tests hierarchical feature extraction versus overfitting.
- **Deep LSTM + Residual Connections:** Tests if skip-connections can rescue deeper recurrent networks from gradient degradation.
- **Shallow GRU:** Evaluates if a simplified cell architecture can match LSTM accuracy with fewer trainable parameters.

---

## 5. Installation & Setup

Follow these steps to clone the repository and run the experiments on your local machine.

### Clone the repository

```bash
git clone https://github.com/yourusername/microsoft-stock-forecasting.git
cd microsoft-stock-forecasting
```

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

- **Activate on Windows**: `venv\Scripts\activate`
- **Activate on Linux/macOS**: `source venv/bin/activate`

### Install Dependencies

```bash
pip install -r requirements.txt
```

(Ensure your `requirements.txt` includes: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter`)

---

## 6. Usage & Configuration

Because the architecture is modular, you do not need to alter the Python code to run new experiments.

1. **Modify Global Settings:** Open `config/config.json` to change the chunk size, batch size, learning rate, or patience.
2. **Design New Architectures:** Open `config/experiments.json` to add new models. You can toggle layer depth, cell type (`LSTM`, `GRU`, `VANILLA`), dropout rates, and residual connections (`true`/`false`).
3. **Execute:** Launch Jupyter Notebook and "Run All" cells in `src/main_notebook.ipynb`.
   ```bash
   jupyter notebook
   ```

---

## 7. Results & Output

Upon completion, the script automatically generates two artifacts in the `results/` folder:

1. `summary\_[TIMESTAMP].json`: A comprehensive metadata file logging the exact architecture configurations, parameter counts (weight footprint), RMSE (USD), and MAPE (%) for every tested model.
2. `stock_predictions_[TIMESTAMP].png`: A high-resolution overlay of all model predictions against the actual Microsoft stock ground-truth line.

# Microsoft Stock Market Time-Series Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Data_Scaling-F7931E)

## 1. Introduction

*Project for the Neural Networks course, University of L'Aquila (June 2026). Individual work.*

This project is a comparative deep learning study focused on financial time-series forecasting. Using historical daily trading data for Microsoft stock (2001–2021), this repository implements, trains, and evaluates a highly modular Neural Network architecture to predict future closing prices.

Rather than deploying a single algorithm, this project evaluates the theoretical strengths, gradient dynamics, and computational efficiencies of various Recurrent Neural Network (RNN) engines, including Vanilla RNNs, LSTMs, and GRUs. It also explores advanced architectural concepts such as Deep Stacking and Residual (Skip) Connections to mitigate gradient degradation in deep temporal networks.

---

## 2. Table of Contents

- [Introduction](#1-introduction)
- [Project Folder Structure](#3-project-folder-structure)
- [Methodology & Models Evaluated](#4-methodology--models-evaluated)
- [Installation & Setup](#5-installation--setup)
- [Usage & Configuration](#6-usage--configuration)
- [Results & Output](#7-results--output)
- [Conclusions](#8-conclusions)

---

## 3. Project Folder Structure

The code is kept separate from the experiment configuration and the data, so new experiments only require editing JSON files.

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
git clone https://github.com/flebo45/Microsoft_Stock_Market_Timeseries_Forecast.git
cd Microsoft_Stock_Market_Timeseries_Forecast
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
3. **Execute:** Launch Jupyter Notebook and "Run All" cells in `src/main.ipynb`.
   ```bash
   jupyter notebook
   ```

---

## 7. Results & Output

Upon completion, the script automatically generates two artifacts in the `results/` folder:

1. `summary_[TIMESTAMP].json`: A comprehensive metadata file logging the exact architecture configurations, parameter counts (weight footprint), RMSE (USD), and MAPE (%) for every tested model.
2. `stock_predictions_[TIMESTAMP].png`: A high-resolution overlay of all model predictions against the actual Microsoft stock ground-truth line.

### Key results (test set: Feb 2018 – Jan 2021, 750 trading days)

Run `results/summary_2026-06-15_16-50-17.json` (20-day window, 128 hidden units, learning rate 0.003):

| Model | Layers | Residual | Parameters | RMSE (USD) | MAPE |
|---|---|---|---|---|---|
| Naive baseline: tomorrow = today | – | – | 0 | **3.15** | **1.37%** |
| Shallow LSTM + residual | 1 | yes | 67k | 10.19 | 4.03% |
| Shallow GRU | 1 | no | 50k | 14.74 | 5.67% |
| Shallow LSTM | 1 | no | 67k | 14.12 | 5.73% |
| Vanilla RNN | 1 | no | 17k | 22.60 | 9.65% |
| Deep LSTM + residual | 4 | yes | 463k | 39.69 | 17.61% |
| Deep LSTM | 3 | no | 331k | 82.85 | 46.09% |
| Super deep LSTM | 6 | no | 728k | 99.27 | 58.69% |

What these numbers say:
- **Depth without residual connections collapses:** 3 and 6 stacked LSTM layers flatline (MAPE 46–59%); residual connections recover part of the gap (17.6% with 4 layers) but shallow networks remain the best.
- **No network beats the naive baseline.** Predicting tomorrow's close as today's close gives a 1.37% MAPE. This is a well-known result for raw daily prices, which behave close to a random walk: the networks learn the trend but not the next-day move. The baseline was added after the study and is computed on the same chronological test split.
- **Results vary between runs** because no random seed is fixed: the same shallow residual LSTM scored 4.03% in this run and 6.18% in `summary_2026-06-16_17-26-41.json`. Rankings between shallow and deep models are stable; small differences between shallow models are not significant.

Next steps: fix the random seeds and average several runs, predict daily returns instead of raw prices, and add exogenous features (volume, market index).

---

## 8. Conclusions

An extensive ablation study and hyperparameter analysis was conducted to evaluate the capabilities and mathematical limits of deep temporal architectures. Throughout the study, we systematically isolated and tested:

- **Vertical Depth:** Residual skip-connections vs. Vanishing Gradients.
- **Optimization Dynamics:** Learning rate sensitivity and Model Collapse.
- **Temporal Memory:** Lookback window lengths (Chunk Sizes).
- **Representational Capacity:** Hidden unit dimensionality (Underfitting vs. Overfitting).

For the comprehensive breakdown of these experiments, side-by-side visual comparisons, and our final theoretical takeaways regarding the limits of raw price extrapolation, please read the detailed [CONCLUSIONS.md](CONCLUSIONS.md) file.

# 🌫️ Air Pollution Forecasting with LSTM

This project uses **deep learning (LSTM)** to predict air pollution levels based on historical weather and pollution data.  
It demonstrates **time-series forecasting** using feature engineering, scaling, and sequence modeling with TensorFlow/Keras.

---

## 📌 Project Overview

Air pollution is a critical urban problem. This project builds a model that learns from:
- Past pollution levels
- Weather parameters (temperature, pressure, wind)
- Time-dependent patterns (lags, rolling statistics)

and predicts future pollution values using a **stacked LSTM neural network**.

---

## 🛠️ Tech Stack

- **Python** 3.x  
- **Libraries:**  
  - `pandas`, `numpy` → Data manipulation  
  - `matplotlib`, `seaborn` → Visualization  
  - `scikit-learn` → Scaling, train/test split, evaluation metrics  
  - `tensorflow.keras` → LSTM model building  

---

## 📂 Dataset

- **Training Dataset:** `LSTM-Multivariate_pollution.csv.zip`  
- **Testing Dataset:** `pollution_test_data1.csv`  

### Features
| Feature | Description |
|--------|-------------|
| pollution | Target variable – Air pollution level |
| dew | Dew point |
| temp | Temperature |
| press | Atmospheric pressure |
| wnd_dir | Wind direction (encoded numerically) |
| wnd_spd | Wind speed |

Additional **engineered features** include:
- **Lag Features:** pollution/temp/press/dew values from previous 1, 2, 3, 6, 12, 24 hours  
- **Rolling Statistics:** mean and std over 12/24-hour windows  
- **Interaction Features:** temperature × pressure, wind × previous pollution  
- **Difference Features:** temp - dew  

---

## 📊 Approach

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical features
   - Generate lag, rolling, and interaction features

2. **Scaling**
   - MinMaxScaler applied to features and target for better convergence

3. **Sequence Preparation**
   - Convert data to overlapping sequences with `TIMESTEP = 24` (past 24 hours)

4. **Model**
   - **Stacked LSTM** with:
     - `LSTM(128)` → BatchNorm → `LSTM(64)` → BatchNorm
     - Dense layers with ReLU activation
     - Dropout for regularization
   - **Loss Function:** Huber loss (robust to outliers)
   - **Optimizer:** Adam (LR = 0.001)

5. **Training**
   - **EarlyStopping** (patience=5) to avoid overfitting
   - **ReduceLROnPlateau** (factor=0.5) for adaptive learning rate
   - **ModelCheckpoint** to save best model

6. **Evaluation**
   - Metrics: MAE, MSE, RMSE, R², MAPE
   - Inverse-transform predictions to original scale for interpretability

---

## 📈 Results (Sample)

| Dataset | MAE | RMSE | R² |
|--------|------|------|----|
| Train | ~0.03 | ~0.04 | ~0.95 |
| Validation | ~0.03 | ~0.04 | ~0.94 |
| Test | ~0.03 | ~0.04 | ~0.92 |

✅ **Low error + High R² = Strong predictive performance**  
✅ **No overfitting observed** (val_loss ≈ train_loss)

---
Next Steps

Try GRU layers for faster training

Experiment with multi-step forecasting (predict next 6–12 hours at once)

Deploy model with Flask/FastAPI for real-time inference

## 🚀 How to Run

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/air-pollution-lstm.git
   cd air-pollution-lstm

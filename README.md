# 📈 Stock Price Prediction using Hybrid Models

This project is a **Streamlit-based web application** that predicts stock prices using a combination of:

- 📊 Time Series Models: ARIMA, SARIMA, Prophet  
- 🤖 Deep Learning Model: LSTM  
- 🧠 AI Analysis: Google Gemini API  
- 📉 Evaluation Metrics: MAE, RMSE, MAPE, R²  

It also generates **AI-based BUY / SELL / HOLD signals** with confidence levels.

---

## 🚀 Features

- 📥 Fetch real-time stock data (Yahoo Finance)
- 📊 Visualize historical stock trends
- 🔮 Forecast future prices using multiple models
- 🏆 Compare model performance
- 🤖 AI-generated trading signals (Gemini)
- 📋 Forecast table + reasoning
- 📉 Moving averages (SMA, EMA)

---

## 🖥️ Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- Statsmodels
- Prophet
- Scikit-learn
- Plotly
- Yahoo Finance API
- Google Gemini API

---

## 📂 Project Structure
```text
stock-prediction/
├── .streamlit/
│   └── secrets.toml       # API keys (Keep this in .gitignore!)
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation Guide (Step-by-Step)

Follow these steps to run this project on **another laptop or PC**:

---

### ✅ Step 1: Install Python

- Download Python (3.9 or higher recommended)
- Install from: https://www.python.org/downloads/
- During installation, check ✅ **"Add Python to PATH"**

---

### ✅ Step 2: Clone the Repository

Open terminal / command prompt:

```bash
git clone https://github.com/your-username/stock-prediction.git
cd stock-prediction
```

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

---

### ✅ Step 3: Create Virtual Environment (Recommended)
```bash
python -m venv venv
```
Activate it:

Windows:
```bash
venv\Scripts\activate
```
Mac/Linux:
```bash
source venv/bin/activate
```

---

###✅ Step 4: Install Required Libraries

Create a requirements.txt file (or use below):
```bash
streamlit
yfinance
pandas
numpy
statsmodels
prophet
scikit-learn
tensorflow
plotly
google-generativeai
```
Install dependencies:
```bash
pip install -r requirements.txt
```

---

###✅ Step 5: Setup Gemini API Key
1. Go to Google AI Studio
2. Generate your Gemini API Key

---

###✅ Step 6: Add API Key to Streamlit Secrets
Create folder:
```bash
.streamlit/
```
Inside it, create file:
```bash
secrets.toml
```
Add this:
```bash
GEMINI_API_KEY = "your_api_key_here"
```

---

###✅ Step 7: Run the Application

```bash
streamlit run app.py
```

---

###✅ Step 8: Open in Browser

Streamlit will automatically open:
```bash
http://localhost:8501
```


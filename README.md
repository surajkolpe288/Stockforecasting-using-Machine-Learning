# 📈 Stock Forecasting using Machine Learning

This project aims to predict future stock price movements using machine learning techniques such as **Linear Regression**, **LSTM (Long Short-Term Memory)**, and **GRU (Gated Recurrent Units)**. It explores the power of deep learning in financial forecasting and addresses challenges like market volatility and noise.

## 📌 Objectives

- Forecast stock prices based on historical data and market indicators.
- Compare the performance of traditional and deep learning models.
- Evaluate model performance using metrics like **RMSE**, **MAE**, and **MAPE**.
- Develop a robust and adaptive system for financial predictions.

## 🧠 Concepts and Methods

- **Data Preprocessing**: Cleaning and normalizing historical stock data.
- **Feature Engineering**: Creating meaningful indicators such as moving averages and volume trends.
- **Model Development**:
  - Linear Regression (baseline)
  - LSTM and GRU for time-series prediction
- **Model Evaluation**: Using RMSE and other metrics to assess accuracy.
- **Model Deployment**: Integration with Flask for a user-friendly web interface.

## 🛠 Tools and Technologies

- `Python`
- `Flask` (Web framework)
- `TensorFlow` (Deep Learning)
- `Scikit-learn`, `Pandas`, `NumPy`
- `yfinance` (Stock data)
- `NewsAPI`, `Alpha Vantage API`, `Finnhub API` (Real-time financial/news data)
- `TextBlob` (Sentiment analysis)
- `Flask-Login` (User authentication)

## 📊 Architecture

1. **Data Collection**
2. **Data Preprocessing**
3. **Model Training & Selection**
4. **Model Evaluation**
5. **Deployment & Visualization**

## 🔍 Results

| Model             | Accuracy (%) |
|------------------|--------------|
| GRU               | 92.5%        |
| LSTM              | 90.8%        |
| Linear Regression | 85.4%        |

GRU performed the best, showing its ability to handle complex time-series patterns and market dynamics effectively.

## 🖥 Interface

A Flask-based dashboard was developed for users to:
- View forecasts
- Access screeners and insights
- Analyze news sentiment


## ✅ Conclusion

Machine learning, especially deep learning models like GRU, offers powerful tools for forecasting stock prices with high accuracy. This project demonstrates the potential of combining financial data, sentiment analysis, and adaptive models in real-world investment decisions.

---


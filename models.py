import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.model_selection import train_test_split
import logging

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO)

# Prepare stock data for modeling
def prepare_data(stock_data):
    try:
        if stock_data.empty:
            raise ValueError("Provided stock data is empty.")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data = stock_data[['Close']].copy()
        scaled_data = scaler.fit_transform(stock_data)
        return scaled_data, scaler
    except Exception as e:
        logging.error(f"Error in prepare_data: {e}")
        raise

# LSTM model
def lstm_model(stock_data):
    try:
        data, scaler = prepare_data(stock_data)
        X, y = [], []
        for i in range(60, len(data)):
            X.append(data[i-60:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential([
            Input(shape=(X.shape[1], 1)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        prediction = model.predict(X[-1].reshape(1, 60, 1), verbose=0)

        if prediction.size == 0:
            raise ValueError("LSTM model returned an empty prediction.")
        
        return scaler.inverse_transform(prediction)[0][0]
    except Exception as e:
        logging.error(f"Error in LSTM model: {e}")
        raise

# GRU model
def gru_model(stock_data):
    try:
        data, scaler = prepare_data(stock_data)
        X, y = [], []
        for i in range(60, len(data)):
            X.append(data[i-60:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential([
            Input(shape=(X.shape[1], 1)),
            GRU(50, return_sequences=True),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        prediction = model.predict(X[-1].reshape(1, 60, 1), verbose=0)

        if prediction.size == 0:
            raise ValueError("GRU model returned an empty prediction.")
        
        return scaler.inverse_transform(prediction)[0][0]
    except Exception as e:
        logging.error(f"Error in GRU model: {e}")
        raise

# Linear regression model
def linear_regression_model(stock_data):
    try:
        if stock_data.empty:
            raise ValueError("Provided stock data is empty.")

        stock_data = stock_data[['Close']].copy()
        stock_data['Date'] = stock_data.index.astype(np.int64) // 10**9

        X = stock_data[['Date']]
        y = stock_data['Close']

        model = LinearRegression()
        model.fit(X, y)
        next_date = stock_data['Date'].max() + 86400  # Add 1 day (in seconds)
        prediction = model.predict([[next_date]])

        if prediction.size == 0:
            raise ValueError("Linear Regression model returned an empty prediction.")
        
        return prediction[0]
    except Exception as e:
        logging.error(f"Error in Linear Regression model: {e}")
        raise

# Sentiment analysis using NewsAPI
def sentiment_analysis(ticker):
    try:
        newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))

        news = newsapi.get_everything(
            q=ticker,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )

        if news['articles']:
            headlines = ' '.join([article['title'] for article in news['articles']])
        else:
            headlines = f"No news found for {ticker}"

        # Perform sentiment analysis
        blob = TextBlob(headlines)
        return blob.sentiment.polarity
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return 0  # Return a neutral sentiment score if there's an error

# Combine all models
def predict_stock(stock_data, ticker):
    try:
        lstm_pred = lstm_model(stock_data)
        gru_pred = gru_model(stock_data)
        linear_pred = linear_regression_model(stock_data)

        return {
            'LSTM': lstm_pred,
            'GRU': gru_pred,
            'Linear Regression': linear_pred
        }
    except Exception as e:
        logging.error(f"Error in stock prediction: {e}")
        return {}

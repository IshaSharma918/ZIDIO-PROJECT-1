import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import math
import streamlit as st


st.title('Stock Market Analysis & Forecasting')
st.subheader('Load your CSV dataset containing Date, Open, High, Low, Close, Volume')


dataset_path = st.file_uploader("Upload CSV", type=["csv"])
if dataset_path is not None:
    data = pd.read_csv(dataset_path, parse_dates=['Date'], index_col='Date')
    st.write('Dataset Preview:', data.head())

    
    data_close = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    
    st.subheader('Closing Price History')
    fig = px.line(data_close, y='Close', title='Stock Closing Price')
    st.plotly_chart(fig)

    
    data_close['50_MA'] = data_close['Close'].rolling(window=50).mean()
    data_close['200_MA'] = data_close['Close'].rolling(window=200).mean()
    fig_ma = px.line(data_close, y=['Close','50_MA','200_MA'], title='Moving Averages')
    st.plotly_chart(fig_ma)

    
    result = adfuller(data_close['Close'])
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])

    
    split = int(len(data_close) * 0.8)
    train, test = data_close['Close'][0:split], data_close['Close'][split:]
    model_arima = ARIMA(train, order=(5,1,0))
    model_arima_fit = model_arima.fit()
    forecast_arima = model_arima_fit.forecast(steps=len(test))

    rmse = math.sqrt(mean_squared_error(test, forecast_arima))
    mae = mean_absolute_error(test, forecast_arima)
    st.write('ARIMA RMSE:', rmse)
    st.write('ARIMA MAE:', mae)

    fig_forecast = px.line()
    fig_forecast.add_scatter(x=train.index, y=train, mode='lines', name='Train')
    fig_forecast.add_scatter(x=test.index, y=test, mode='lines', name='Test')
    fig_forecast.add_scatter(x=test.index, y=forecast_arima, mode='lines', name='ARIMA Forecast')
    st.plotly_chart(fig_forecast)

    
    prophet_data = data_close['Close'].reset_index().rename(columns={'Date':'ds','Close':'y'})
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(prophet_data)
    future = model_prophet.make_future_dataframe(periods=30)
    forecast_prophet = model_prophet.predict(future)
    fig_prophet = px.line(forecast_prophet, x='ds', y='yhat', title='Prophet Forecast')
    st.plotly_chart(fig_prophet)

    
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data_close[['Close']])

    train_size = int(len(data_scaled) * 0.8)
    train_lstm, test_lstm = data_scaled[0:train_size], data_scaled[train_size:]

    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 50
    X_train, Y_train = create_dataset(train_lstm, time_step)
    X_test, Y_test = create_dataset(test_lstm, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
    model_lstm.add(LSTM(50))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    model_lstm.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=5, batch_size=32, verbose=0)

    train_predict = model_lstm.predict(X_train)
    test_predict = model_lstm.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    plt.figure(figsize=(12,6))
    plt.plot(data_close['Close'].values, label='Actual')
    plt.plot(np.arange(time_step, len(train_predict)+time_step), train_predict, label='Train Predict')
    plt.plot(np.arange(len(train_predict)+(time_step*2)+1, len(data_scaled)-1), test_predict, label='Test Predict')
    plt.title('LSTM Forecast')
    plt.legend()
    st.pyplot(plt)

    st.write('Use this dashboard to visualize stock price trends and forecasts using ARIMA, Prophet, and LSTM models.')
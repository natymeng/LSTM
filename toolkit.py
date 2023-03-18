import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

def load_data():
    stocks = pd.read_csv('stock.csv')
    stocks['Date'] = pd.to_datetime(stocks['Date'])
    stocks['Date'] = stocks['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    
    fx = pd.read_csv('fx.csv')
    classification = pd.read_csv('classification.csv')
    
    stocks = stocks.merge(classification, how = 'left')
    stocks = stocks.merge(fx, how = 'left', on = 'Date')
    stocks['Price USD'] = np.where(stocks['Country'] == 'JP', stocks['Price']/stocks['JP/US'], 
                              np.where(stocks['Country'] == 'UK', stocks['Price'] * stocks['US/UK'], stocks['Price']))
    return stocks

def stock_df(stocks, ticker):
    df = stocks[stocks['ID'] == ticker][['Date','Country','Volume','Volatility','Price USD','Sentiment','Spread']]
    df.fillna(method = 'ffill', inplace = True) #in case there's no data point, fill with previous day data point
    return df

def normalized_data(data_series, feature_low=0, feature_high=1):
    scaler = MinMaxScaler(feature_range=(feature_low,feature_high)) 
    scaler.fit(data_series.values.reshape(-1, 1))
    data_norm = scaler.transform(data_series.values.reshape(-1, 1))
    return data_norm

def df_to_XY(px_norm, window_size=5):
    """
    Takes a normalized series and a rolling window size and output 2 arrays of X and Y for training and testing.
    The default window size is 5.
    """
    X = []
    Y = []
    for i in range(len(px_norm)-window_size):
        X.append([pair for pair in px_norm[i:i+window_size]])
        Y.append(px_norm[i+window_size, 0])
    return np.array(X), np.array(Y)

def split_data(X, Y, train_split = 0.8): #default train/test split is 80/20
    """
    Takes 2 arrays of normalized data X and Y and split the data into training and rest set.
    The default split is 80/20.
    """
    split_index = int(Y.shape[0] * train_split)
    
    X_train, Y_train = X[:split_index], Y[:split_index]
    X_test, Y_test = X[split_index:], Y[split_index:]
    
    return X_train, Y_train, X_test, Y_test

def performance_measure(model, X, y, method = 'Standard LSTM'):
    """
    Takes the setup model, X, Y and the method name to create a df of performance measure. 
    The df includes MSE, MAE, MAPE and Accuracy (1-MAPE).
    """
    predictions = model.predict(X).flatten()
    return pd.DataFrame({'MSE': mse(y, predictions), 
                         'MAE': mae(y, predictions), 
                         'MAPE': mape(y, predictions), 
                         'Accuracy': 1-mape(y, predictions)}, index = [method])

def LSTM_layers(input_row, input_col):
    """
    Takes an input row number and an column number to set up the layers in the various versions of LSTM models.
    """
    model0 = Sequential()
    model0.add(LSTM(units=100,return_sequences=True,input_shape=(input_row, input_col)))
    model0.add(LSTM(units=100,return_sequences=True))
    model0.add(Dropout(0.1))
    model0.add(LSTM(units=100,return_sequences=True))
    model0.add(Dropout(0.1))
    model0.add(LSTM(units=100))
    model0.add(Dropout(0.1))
    model0.add(Dense(units=1))
    return model0

def standard_LSTM(data_series, feature_low = 0, feature_high = 1, window_size = 5, train_split = 0.8, method = 'Standard LSTM'):
    """
    This function sets up the standard LSTM model.
    This model has 3 LSTM layer with 100 units per layer, followed by a dropout layer with dropout rate of 0.1.
    The input later takes the input shape of nx1, where n is the size of the rolling window.
    """
    px_norm = normalized_data(data_series, feature_low, feature_high)
    X0, Y0 = df_to_XY(px_norm, window_size)
    X0_train, Y0_train, X0_test, Y0_test = split_data(X0, Y0, train_split)

    model0= LSTM_layers(window_size, 1)
    model0.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
    model0.fit(X0_train,Y0_train, epochs=10, verbose = 0)

    res0 = performance_measure(model0, X0_test, Y0_test.flatten(), method)

    return model0, res0

def LSTM_pair(data_series1, data_series2, feature_low1 = 0, feature_high1 = 1, feature_low2 = 0, feature_high2 = 1, 
              window_size = 5, train_split = 0.8, method = 'Standard LSTM'):
    """
    This function sets up the LSTM model with historical price data and sentiment score data as a pair.
    This model has 3 LSTM layer with 100 units per layer, followed by a dropout layer with dropout rate of 0.1.
    The input later takes the input shape of nx2, where n is the size of the rolling window.
    """
    px_norm = normalized_data(data_series1, feature_low1, feature_high1)
    sent_norm = normalized_data(data_series2, feature_low2, feature_high2)
    df_norm = np.hstack((px_norm[1:], sent_norm[:-1]))
    
    X0, Y0 = df_to_XY(df_norm, window_size)
    X0_train, Y0_train, X0_test, Y0_test = split_data(X0, Y0, train_split)

    model0= LSTM_layers(window_size, 2)
    model0.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
    model0.fit(X0_train,Y0_train, epochs=10, verbose = 0)

    res1 = performance_measure(model0, X0_test, Y0_test.flatten(), method)
    return model0, res1

def df_to_XY_add_sent(px_norm, sent_norm, window_size=5):
    X = []
    Y = []
    for i in range(len(px_norm)-window_size):
        X.append(np.concatenate((px_norm[i:i+window_size], sent_norm[i+window_size-1:i+window_size]), axis = 0))
        Y.append(px_norm[i+window_size, 0])
    return np.array(X), np.array(Y)

def LSTM_addl_feature(data_series1, data_series2, feature_low1 = 0, feature_high1 = 1, feature_low2 = 0, feature_high2 = 1, 
                      window_size = 5, train_split = 0.8, method = 'Standard LSTM'):
    """
    This function sets up the LSTM model with historical price data and sentiment score data as an additional feature on the input.
    This model has 3 LSTM layer with 100 units per layer, followed by a dropout layer with dropout rate of 0.1.
    The input later takes the input shape of (n+1)x1, where n is the size of the rolling window.
    """
    px_norm = normalized_data(data_series1, feature_low1, feature_high1)
    sent_norm = normalized_data(data_series2, feature_low2, feature_high2)

    df_norm = np.hstack((px_norm[1:], sent_norm[:-1]))
    
    X0, Y0 = df_to_XY_add_sent(px_norm, sent_norm, window_size)
    X0_train, Y0_train, X0_test, Y0_test = split_data(X0, Y0, train_split)
    
    model0= LSTM_layers(window_size+1, 1)
    model0.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
    model0.fit(X0_train,Y0_train, epochs=10, verbose = 0)

    res2 = performance_measure(model0, X0_test, Y0_test.flatten(), method)    
    return model0, res2


# def compare(res0, res1, res2):
#     return pd.concat([res0, res1, res2], axis = 0)

def run_price_predict(ticker):
    df_stock = stock_df(load_data(), ticker)
    df1 = df_stock['Price USD']
    df2 =  df_stock['Sentiment']
    summary = pd.concat([standard_LSTM(df1)[1], 
                     LSTM_pair(df1,df2, method = 'LSTM Pair')[1],
                     LSTM_addl_feature(df1, df2, method = 'LSTM Addl feature')[1]])
    return summary


def predict_LSTM_af(df, window_size = 5, train_split = 0.8):
    """
    This function uses the LSTM Additional Feature model to predict the one day ahead stock price.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(df['Sentiment'].values.reshape(-1, 1))
    sent_norm = scaler.transform(df['Sentiment'].values.reshape(-1, 1))

    scaler.fit(df['Price USD'].values.reshape(-1, 1)) # fit the price after sentiment so that I can inverse transform the price back later
    px_norm = scaler.transform(df['Price USD'].values.reshape(-1, 1))

    df_norm = np.hstack((px_norm[1:], sent_norm[:-1]))

    X0, Y0 = df_to_XY_add_sent(px_norm, sent_norm, window_size)
    X0_train, Y0_train, X0_test, Y0_test = split_data(X0, Y0, train_split)

    model0= LSTM_layers(window_size+1, 1)
    model0.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
    model0.fit(X0_train,Y0_train, epochs=10, verbose = 0)

    oneday_predict = model0.predict(np.append(px_norm[-window_size:], sent_norm[-1:], axis=0).reshape(1, window_size+1, 1))
    oneday_predict_px = scaler.inverse_transform(oneday_predict)
    accuracy = performance_measure(model0, X0_test, Y0_test.flatten())['Accuracy']

    return oneday_predict_px[0][0], accuracy[0]

    
def predivt_LSTM_volume(ticker, window_size = 5, train_split = 0.8):
    df = stock_df(load_data(), ticker).set_index('Date')
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(df['Volume'].values.reshape(-1, 1))
    vol_norm = scaler.transform(df['Volume'].values.reshape(-1, 1))

    window_size = 5
    train_split = 0.8

    X, Y = df_to_XY(vol_norm, window_size=5)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, train_split)

    model3 = LSTM_layers(window_size, 1)
    model3.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
    model3.fit(X_train,Y_train, epochs=5, verbose = 0)

    oneday_predict = model3.predict(vol_norm[-window_size:].reshape(1, window_size, 1))
    oneday_predict_volume = scaler.inverse_transform(oneday_predict)

    return oneday_predict_volume
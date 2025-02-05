import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

class ModelLSTM:
    def __init__(self, df, starting_col_ind, target_feature, test_size, visualization_obj):
        self.df = df
        self.starting_col_ind = starting_col_ind
        self.target_feature = target_feature
        self.test_size = test_size
        self.model = None
        self.visualization = visualization_obj
    def create_train_test(self):
        X = self.df.iloc[:,self.starting_col_ind:].values
        y = self.df[self.target_feature]
        X = X.reshape((X.shape[0], X.shape[1], 1)) 
        X_train = X[:-self.test_size]
        y_train = y[:-self.test_size]
        X_test = X[-self.test_size:]
        y_test = y[-self.test_size:]
        return X_train, y_train, X_test, y_test
    def predict_close_LSTM(self, X_train, y_train, X_test, y_test):
        model=Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping])
        self.model = model
        self.visualization.plot_history(history)
        return history
    def run_all(self):
        X_train, y_train, X_test, y_test = self.create_train_test()
        history = self.predict_close_LSTM(X_train, y_train, X_test, y_test)
        self.visualization.update(self.model, history, X_train, y_train, X_test, y_test)
        y_test, y_pred = self.visualization.print_result()
        return y_test, y_pred

class ModelLSTMClose:
    def __init__(self, df, target_feature, test_size, window_size, visualization_obj):
        self.df = df
        self.target_feature = target_feature
        self.test_size = test_size
        self.window_size = window_size
        self.model = None
        self.visualization = visualization_obj
    def create_train_test(self):
        train_data = self.df[self.target_feature][:-self.test_size]
        train_data = train_data.values.reshape(-1,1)
        test_data = self.df[self.target_feature][-(self.test_size+self.window_size):]
        test_data = test_data.values.reshape(-1,1)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for i in range(self.window_size, len(train_data)):
            X_train.append(train_data[i-self.window_size:i, 0])
            y_train.append(train_data[i, 0])
        for i in range(self.window_size, len(test_data)):
            X_test.append(test_data[i-self.window_size:i, 0])
            y_test.append(test_data[i, 0])        
        X_train = np.array(X_train)
        X_test  = np.array(X_test)
        y_train = np.array(y_train)
        y_test  = np.array(y_test)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_train = np.reshape(y_train, (-1,1))
        y_test  = np.reshape(y_test, (-1,1))
        print('X_train Shape: ', X_train.shape)
        print('y_train Shape: ', y_train.shape)
        print('X_test Shape:  ', X_test.shape)
        print('y_test Shape:  ', y_test.shape)
        return X_train, y_train, X_test, y_test
    def predict_close_LSTM(self, X_train, y_train, X_test, y_test):
        model=Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(self.window_size, 1)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping])
        self.model = model
        self.visualization.plot_history(history)
        return history
    def run_all(self):
        X_train, y_train, X_test, y_test = self.create_train_test()
        history = self.predict_close_LSTM(X_train, y_train, X_test, y_test)
        self.visualization.update(self.model, history, X_train, y_train, X_test, y_test)
        y_test, y_pred = self.visualization.print_result()
        return y_test, y_pred

def lgb_model(df, target_feature, window_size, test_size):
    train_data = df[target_feature][:-test_size]
    train_data = train_data.values.reshape(-1,1)
    test_data = df[target_feature][-(test_size+window_size):]
    test_data = test_data.values.reshape(-1,1)
    X_train_lgb = []
    y_train_lgb = []
    for i in range(window_size, len(train_data)):
        X_train_lgb.append(train_data[i-window_size:i, 0])
        y_train_lgb.append(train_data[i, 0])
    X_test_lgb = []
    y_test_lgb = []
    for i in range(window_size, len(test_data)):
        X_test_lgb.append(test_data[i-window_size:i, 0])
        y_test_lgb.append(test_data[i, 0])
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
    lgb_model.fit(X_train_lgb, y_train_lgb)
    y_pred_lgb = lgb_model.predict(X_test_lgb)
    y_pred_train_lgb = lgb_model.predict(X_train_lgb)
    
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train_lgb, y_pred_train_lgb))}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test_lgb, y_pred_lgb))}") 
    y_test_lgb = np.array(y_test_lgb)
    y_test_lgb = y_test_lgb.reshape(-1,1)
    
    y_pred_lgb = np.array(y_pred_lgb)
    y_pred_lgb = y_pred_lgb.reshape(-1,1)
    MAPE = mean_squared_error(y_test_lgb, y_pred_lgb)
    Accuracy = 1 - MAPE
    print(f"Accuracy : {Accuracy}")
    return lgb_model, y_test_lgb, y_pred_lgb


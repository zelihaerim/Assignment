import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
from visualization.visualization import Visualization
# create new features
def calculate_rsi(df, timeperiod=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=timeperiod, min_periods=1).mean()
    avg_loss = loss.rolling(window=timeperiod, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def calculate_ema(df, column, period):
    alpha = 2 / (period + 1)
    ema = df[column].ewm(span=period, adjust=False).mean()
    return ema
def create_RSI_MACD_BB(df):
    # RSI hesaplama
    # '14' periyot için RSI
    df['RSI'] = calculate_rsi(df) # ta.RSI(df['Close'], timeperiod=14)
    # MACD hesaplama
    # MACD, Signal line ve Histogram için periyotlar
    # df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    fastperiod=12
    slowperiod=2
    signalperiod=9
    ema_fast = calculate_ema(df, 'Close', fastperiod)
    ema_slow = calculate_ema(df, 'Close', slowperiod)
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = calculate_ema(df, 'MACD', signalperiod)
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    # Bollinger Bands hesaplama
    # '20' periyot ve '2' standart sapma ile Bollinger Bands hesaplama
    # df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    timeperiod = 20
    nbdevup = 2
    nbdevdn = 2
    df['SMA'] = df['Close'].rolling(window=timeperiod).mean()
    df['STD'] = df['Close'].rolling(window=timeperiod).std()
    df['BB_upper'] = df['SMA'] + (df['STD'] * nbdevup)
    df['BB_lower'] = df['SMA'] - (df['STD'] * nbdevdn)
    df['BB_middle'] = df['SMA']
    df.drop(columns=['SMA', 'STD'], inplace=True)
    # Sonuçları gösterme
    print(df[['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower']].tail())
# impute nan values
def find_test_size(df, year):# gold_df
    test_size = df[df.Date.dt.year>=year].shape[0]
    print(f"test size: {test_size}")
    return test_size

# Impute nan values class
class ImputeClass:
    def __init__(self, df:pd.DataFrame, window_size_lst:list, visualization_obj, test_size):
        self.will_be_imputed = df.columns[df.isnull().any()].tolist()
        self.df = df
        self.test_size = test_size
        self.window_size_lst = window_size_lst
        self.missing_count = 0
        self.visualization = visualization_obj
    # All below 4 functions are helper functions
    def find_feature_index(self, df, feature_name):
        for ind, i in enumerate(df.columns):
            if(i == feature_name):
                return ind
        return -1
    def start_impute_last_index(self, scaled_df, target_column_index):
        model_will_impute = scaled_df[scaled_df.iloc[:,target_column_index].isnull()].iloc[:, target_column_index]
        return model_will_impute.index[-1]    
    def define_model(self, early_stopping, X_train, y_train, X_test, y_test, window_size_=100): # helper
        model=Sequential()
        model.add(LSTM(32,return_sequences=True,input_shape=(window_size_,1)))
        model.add(LSTM(64,return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer='adam')
        print(model.summary())
        history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping])
        return model, history

    def run_model(self, X_train, y_train, X_test, y_test, window_size): # helper
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        print(f"window_size: {window_size}")
        model, history = self.define_model(early_stopping, X_train, y_train, X_test, y_test, window_size)
        model.save('lstm_model_'+str(self.missing_count)+'.h5')
        self.missing_count += 1
        self.visualization.plot_history(history)
        return model, history
    def scale_df_fun(self): # gold_df
        scaler = MinMaxScaler()
        scaled_gold_df = scaler.fit_transform(self.df.iloc[: ,1:])
        scaled_df = pd.DataFrame(data=scaled_gold_df, columns=self.df.columns[1:])
        return scaled_df
    def impute_column(self, df, model, missing_value_count, window_size, feature_name): # helper
        for i in range(0, missing_value_count+1):
            impute_set = df[feature_name][(window_size-i):2*window_size-i].values[::-1]
            impute_set = impute_set.reshape(1,-1,1)
            result = model.predict(impute_set)
            print(result)
            df[feature_name].loc[(window_size-1)-i]=result[0][0]
        return df
    def train_test_split(self, scaled_df, target_feature_name, window_size):
        feature_index = self.find_feature_index(scaled_df, target_feature_name)
        if(feature_index == -1):
            raise ValueError("feature_index can not be -1, Column must be deleted.")
        train_data = scaled_df[target_feature_name][:-self.test_size]
        train_data = train_data[(self.start_impute_last_index(scaled_df, feature_index))+1:].values[::-1].reshape(-1,1)
        X_train = []
        y_train = []
        for i in range(window_size, len(train_data)):
            X_train.append(train_data[i-window_size:i, 0])
            y_train.append(train_data[i, 0])
        test_data = scaled_df[target_feature_name][-(self.test_size+window_size):]
        test_data = test_data.values[::-1].reshape(-1,1)
        X_test = []
        y_test = []
        for i in range(window_size, len(test_data)):
            X_test.append(test_data[i-window_size:i, 0])
            y_test.append(test_data[i, 0])
        # convert array
        X_train = np.array(X_train)
        X_test  = np.array(X_test)
        y_train = np.array(y_train)
        y_test  = np.array(y_test)
        # reshape
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_train = np.reshape(y_train, (-1,1))
        y_test  = np.reshape(y_test, (-1,1))
        # print info
        print('X_train Shape: ', X_train.shape)
        print('y_train Shape: ', y_train.shape)
        print('X_test Shape:  ', X_test.shape)
        print('y_test Shape:  ', y_test.shape)
        return X_train, y_train, X_test, y_test
   
    def run_all(self, window_size_lst:list): # main function
        scaled_df = self.scale_df_fun()
        for ind, target_feature_name in enumerate(self.will_be_imputed):
            window_size = window_size_lst[ind]
            X_train, y_train, X_test, y_test = self.train_test_split(scaled_df, target_feature_name, window_size)
            model, history = self.run_model(X_train, y_train, X_test, y_test, window_size)
            self.visualization.update(model, history, X_train, y_train, X_test, y_test)
            self.visualization.print_result()
            missing_value_count =  scaled_df[target_feature_name].isnull().sum()
            scaled_df = self.impute_column(scaled_df, model, missing_value_count, window_size, target_feature_name)
        return scaled_df

def run_impute(gold_df):
    window_size_lst = [] 
    for i in gold_df.columns[gold_df.isnull().any()].tolist():
        window_size_lst.append(gold_df[i].isnull().sum())
    print(f"window_size_lst : {window_size_lst}")
    try:
        visualization_obj = Visualization()
        impute_obj = ImputeClass(gold_df, window_size_lst, visualization_obj, test_size = find_test_size(gold_df, 2018))
        filled_df = impute_obj.run_all(window_size_lst)
        return filled_df
    except ValueError:
        print("Column is not in df anymore.")
    except:
        print("Something is wrong.")

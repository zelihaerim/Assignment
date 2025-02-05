import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
#%matplotlib inline
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Line chart - historical view
def plot_historical_line(df, features, date_feature_name):
    nrow = int(np.ceil(len(features)/2))
    fig, axs = plt.subplots(nrow, 2, figsize=(15, 15))
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        df.set_index(date_feature_name)[feature].plot(ax=axs[row, col], color="orange", lw=1.5)
        axs[row, col].set_ylabel(feature)
        axs[row, col].set_xlabel("Date")   
    if(len(features)%2 == 1):
        axs[(nrow-1), 1].axis('off')
    plt.tight_layout()
    plt.show()

def plot_historical_line_one(df, features, date_feature_name):
    num_colors = len(features)
    colors = cm.viridis(np.linspace(0, 1, num_colors))
    plt.figure(figsize=(15, 8))
    for i, feature in enumerate(features):
        df.set_index(date_feature_name)[feature].plot(color=colors[i], label=feature, lw=1.5)
    plt.ylabel('Values')
    plt.xlabel('Date')
    plt.legend(loc='upper right')
    plt.show()
# Line chart - historical view
def plot_ma_line(df, features, date_feature_name:str, ma_day:list):
    if(len(features)==0 or len(ma_day)==0 or len(df)==0):
        return
    if(not set(features).issubset(df.columns)):
        return
    nrow = int(np.ceil(len(features)/2))
    ncol = 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(40, 10))
    num_colors = len(ma_day)
    colors = cm.autumn_r(np.linspace(0, 1, num_colors)) # Blues
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        ma_df=pd.DataFrame(data=df[date_feature_name], columns=[date_feature_name])
        df.set_index(date_feature_name)[feature].plot(ax=axs[row, col], color="orange", label=feature, lw=1)
        for ind, ma in enumerate(ma_day):
            column_name = f"MA_for_{ma}_days"
            column_name = column_name+f"_{feature}"
            ma_df[column_name] = df[feature].rolling(ma).mean()
            ma_df[column_name].plot(ax=axs[row, col], color=colors[ind], label=column_name, lw=2)
            axs[row, col].set_ylabel(feature)
            axs[row, col].set_xlabel("Date")   
            axs[row, col].legend(loc='lower right')
    if(len(features)%2 == 1 and len(features) != 1):
        axs[(nrow-1), 1].axis('off')
        print("No")
    plt.tight_layout()
    plt.ylabel('Values')
    plt.xlabel('Date')
    plt.show()
# Plot Daily Return
def plot_daily_return(df, close_feature_name, date_feature_name):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Daily_Return'] = df['Close'].pct_change()
    average_daily_return = df['Daily_Return'].mean()
    print("Avarage Daily Return : ", average_daily_return)
    plt.figure(figsize=(15, 8))
    df.set_index(date_feature_name)['Daily_Return'].plot(color="orange", label=close_feature_name, lw=1.5,
                                                        linestyle='--', marker='o')
    plt.ylabel('Values')
    plt.xlabel('Date')
    plt.legend(loc='upper right')
    plt.show()
    return average_daily_return
# Histogram Daily Return
def plot_hist(df, feature_name):
    plt.figure(figsize=(15, 5))
    df[feature_name].hist(bins=50)
    plt.xlabel(feature_name)
    plt.ylabel('Counts')
    plt.ylim()
    plt.title(f'Gold')    
    plt.tight_layout()

# after prediction visualization
class Visualization:
    def __init__(self, model= None, history= None, X_train= None, y_train= None, X_test= None, y_test= None):
        self.model = model
        self.history = history
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def update(self, model, history, X_train, y_train, X_test, y_test):
        if model is not None:
            self.model = model
        if history is not None:
            self.history = history
        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train
        if X_test is not None:
            self.X_test = X_test
        if y_test is not None:
            self.y_test = y_test
    def plot_history(self, history): # helper
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Test Loss')
        plt.title('Train and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.legend()
        plt.show()
    def print_prediction_result(self, model, X, y):
        result = model.evaluate(X, y)
        y_pred = model.predict(X)
        MAPE = mean_squared_error(y, y_pred)
        Accuracy = 1 - MAPE
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print("Loss:", np.round(result, 10))
        print("MAPE:", np.round(MAPE, 10))
        print("Accuracy:", Accuracy)
        print(f"RMSE: {rmse}")
        return y, y_pred
    def print_result(self):
        print("Train Result")
        self.print_prediction_result(self.model, self.X_train, self.y_train)
        print("=*"*50)
        print("Test Result")
        y_test, y_pred = self.print_prediction_result(self.model, self.X_test, self.y_test)
        return y_test, y_pred

def plot_prediction(df, test_size, date_feature_name, target_feature_name, y_test, y_pred, scaler):
    y_test_true = scaler.inverse_transform(y_test)
    y_test_pred = scaler.inverse_transform(y_pred)
    train = scaler.inverse_transform(df[target_feature_name].values.reshape(-1,1))
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes',edgecolor='blue')
    plt.plot(df[date_feature_name][:-test_size].values, train[:-test_size], color='orange', lw=2) 
    plt.plot(df[date_feature_name][-test_size:].values, y_test_true.reshape(-1), color='red', lw=2)
    plt.plot(df[date_feature_name][-test_size:].values, y_test_pred.reshape(-1), color='yellow', lw=2)
    plt.title('Model Performance on Gold Price Prediction', fontsize=15)
    plt.xlabel(date_feature_name, fontsize=12)
    plt.ylabel(target_feature_name, fontsize=12)
    plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})
    plt.grid(color='gray')
    plt.show()

def plot_train_test(df, test_size, date_feature_name, target_feature_name):
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes',edgecolor='yellow')
    plt.plot(df[date_feature_name][:-test_size].values, df[target_feature_name][:-test_size].values, color='#d35400', lw=2)
    plt.plot(df[date_feature_name][-test_size:].values, df[target_feature_name][-test_size:].values, color='#f4d03f', lw=2)
    plt.title('Gold Price Training and Test Sets', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close', fontsize=12)
    plt.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 15})
    plt.grid(color='pink')
    plt.show()

def plot_train_test_window(df, test_size, date_feature_name, target_feature_name):
    window_size = [30, 50, 100, 200, 300, 400, 500]
    for i in window_size:
        plt.figure(figsize=(30, 6), dpi=150)
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rc('axes',edgecolor='yellow')
        counter=0
        train_color = ["green", "red"]
        test_color = ["purple", "blue"]
        for j in range(0, len(df)-1, i):
            plt.plot(df[date_feature_name][:-test_size][j:j+i].values, df[target_feature_name][:-test_size][j:j+i].values, color=train_color[counter%2], lw=2)
            plt.plot(df[date_feature_name][-test_size:][j:j+i].values, df[target_feature_name][-test_size:][j:j+i].values, color=test_color[counter%2], lw=2)
            counter += 1
            
        plt.title(f'Gold Price Training and Test Sets Window Size = {i}', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close', fontsize=12)
        plt.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 15})
        # Grid ekleme (her 100. veri noktasında grid çizgisi olacak şekilde)
        plt.grid(True, which='both', axis='x', linestyle='--', color='gray', linewidth=0.5)
        plt.show()

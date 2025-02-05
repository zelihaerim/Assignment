import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization.visualization import *
from feature_engineering.feature_engineering import *
from model.model import *

def main():
    # read csv file
    file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'gold.csv')
    gold_df = pd.read_csv(file_path)
    print(gold_df.tail())
    print(gold_df.isnull().sum())
    print(gold_df.info())
    print(gold_df.nunique())
    print(f"is there any duplicate row : {gold_df.duplicated().any()}")
    print(gold_df[gold_df.duplicated()])
    gold_df.drop(columns=["Currency"], inplace=True)
    print(gold_df.describe())
    # create visualization
    plot_historical_line(gold_df, gold_df.columns[1:], "Date")
    plot_historical_line_one(gold_df, gold_df.columns[1:-1], "Date")
    # mechanism of day
    ma_day = [15, 30, 60, 90, 180, 360]
    plot_ma_line(gold_df, gold_df.columns[1:-1], "Date", ma_day)
    ma_day = [15, 30, 60]
    plot_ma_line(gold_df, gold_df.columns[1:-1], "Date", ma_day)
    column_name = f"MA_for_{60}_days"
    column_name = column_name+"_Close"
    gold_df[column_name] = gold_df["Close"].rolling(60).mean()
    plot_daily_return(gold_df, 'Close', 'Date')
    plot_hist(gold_df, 'Daily_Return')
    print(f"Expected return: {gold_df['Daily_Return'].mean()} \nRisk: {gold_df['Daily_Return'].std()}")
    create_RSI_MACD_BB(gold_df)
    filled_df = run_impute(gold_df)
    filled_df["Date"] = gold_df["Date"]
    filled_df = filled_df[[filled_df.columns[-1]] + list(filled_df.columns[:-1])]
    # Plot imputed features
    plot_historical_line_one(filled_df, filled_df.columns[6:], "Date")
    test_size = find_test_size(gold_df, 2018)
    target_feature_name="Close"
    visualization_obj = Visualization()
    model_obj = ModelLSTM(filled_df, 6, target_feature_name, test_size, visualization_obj)
    y_test, y_pred = model_obj.run_all()
    # Plot first approach result
    date_feature_name="Date"
    target_feature_name="Close"
    scaler = MinMaxScaler()
    scaler.fit_transform(gold_df[[target_feature_name]].values.reshape(-1,1))
    visualization_df = pd.concat([filled_df[[date_feature_name, target_feature_name]], filled_df.iloc[:,6:]], axis=1)
    plot_prediction(visualization_df, model_obj.test_size, date_feature_name, target_feature_name, y_test.values.reshape(-1,1), y_pred, scaler)
    # Second approach
    plot_train_test(gold_df, test_size, "Date", "MA_for_60_days_Close")
    plot_train_test_window(gold_df[:2000], test_size, "Date", "Close")
    plot_train_test_window(gold_df[:2000], test_size, "Date", "MA_for_60_days_Close")
    # result 100 is selected as a good window size
    window_size = 100
    target_feature_name = "Close"
    model_close_obj = ModelLSTMClose(pd.DataFrame(filled_df[target_feature_name]), target_feature_name, test_size, window_size, visualization_obj)
    y_test_close, y_pred_close = model_close_obj.run_all()
    # Plot second approach result
    date_feature_name="Date"
    target_feature_name="Close"
    scaler = MinMaxScaler() # create scaler
    scaler.fit_transform(gold_df[[target_feature_name]].values.reshape(-1,1))
    visualization_df = filled_df[[date_feature_name, target_feature_name]]
    plot_prediction(visualization_df, model_close_obj.test_size, date_feature_name, target_feature_name, y_test_close.reshape(-1,1), y_pred_close, scaler)
    # 3 Lightgbm approach
    my_lgb_model, y_test_lgb, y_pred_lgb = lgb_model(filled_df, target_feature_name, window_size, test_size= find_test_size(filled_df, 2018))
    plot_prediction(filled_df, test_size, date_feature_name, target_feature_name, y_test_lgb, y_pred_lgb, scaler)

if __name__ == "__main__":
    main()

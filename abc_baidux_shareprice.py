import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# Load and prepare data
close_price = pd.read_excel('abc_close_price.xlsx', index_col='Date', parse_dates=True)
raw_data = pd.read_excel('乡村振兴_全国.xlsx', index_col='时间', parse_dates=True)
lags = [90, 120, 150, 180, 210]
target_data_column = '搜索pc+移动'
stock_code = '1288.HK'

# Function to create lagged rolling mean columns for training and future prediction
def create_lagged_rolling(df, column, lags, window=30):
    lagged_dfs = []
    for lag in lags:
        # Create lagged rolling mean for training data
        lagged_df = df[[column]].shift(lag).rolling(window).mean()
        lagged_df.columns = [f'{lag}days_lag']
        lagged_dfs.append(lagged_df)
    # Combine all lagged features
    lagged_features = pd.concat(lagged_dfs, axis=1).dropna()
    return lagged_features


def mean_directional_accuracy(y_true, y_pred):
    # Calculate the direction of change for both actual and predicted values
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    # Compare the direction of changes between actual and predicted
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy


def plot_predictions(y_test, y_predict, future_dates=None, future_predictions=None):
    plt.figure(figsize=(12, 6))
    # plt.plot(df.index, y, label='Actual Close Price')
    plt.plot(y_test.index, y_predict, label='Test Predictions', linestyle='--')
    plt.plot(y_test.index, y_test, label='Actual Prices')
    if future_dates is not None:
        plt.plot(future_dates, future_predictions, label='Future Predictions (Next 90 Days)', linestyle='--', color='red')
        plt.title('Actual, Test, and Future Predictions for the Next 90 Days')
    else:
        plt.title('Actual, Test Predictions')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def train_and_backtest_lag(raw_data, target_data_column,
                           stock_code, lags: list, test_size=0.2,
                           n_estimators=300, learning_rate=0.1):
    # Define lag periods and create lagged columns for training

    raw_lagged = create_lagged_rolling(raw_data, target_data_column, lags)

    # Concatenate DataFrames and drop rows with missing values for training data
    df = pd.concat([close_price, raw_lagged], axis=1).dropna()

    # Define features (X) and target (y) for training
    X = df[[f'{lag}days_lag' for lag in lags]]
    y = df[stock_code]

    # Train/test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    model.fit(X_train, y_train)

    # Predict on test set and calculate R-squared score
    y_predict = model.predict(X_test)

    mda = mean_directional_accuracy(y_test, y_predict)
    print(f'Mean Directional Accuracy: {mda:.2f}')
    # 一直在试不同的Metrics, 包括MSPE, RMSE， MAPE，但视觉上更佳的[90, 120, 150, 180, 210]组合始终达不到最优结果
    # 可能是因为尽管该组合对方向判断比较准确，但与真实数据一直保持距离，所以比不过那些对trend不敏感但始终处于真实数据附近的metrics
    # TODO: 找到一个最合适的metrics 对结果衡量

    plot_predictions(y_test, y_predict)
    return model, y_test, y_predict

model, y_test, y_predict = train_and_backtest_lag(raw_data, target_data_column, stock_code, lags)
# -----------------------------------------------



def make_forecast(model, y_test, y_predict, target_data_column, future_periods=lags[0]):
    # Step 2: Prepare data for future predictions
    # Define future date range
    future_dates = pd.date_range(start=raw_data.index[-1] + pd.Timedelta(days=1), periods=future_periods, freq='D')

    # Create a placeholder for future dates with the last known value for simplicity
    placeholder_for_future_data = pd.DataFrame(index=future_dates, columns=raw_data.columns)
    placeholder_for_future_data[target_data_column] = raw_data[target_data_column].iloc[-1]  # Assuming last value as a proxy

    # Concatenate original and future data for lagged rolling calculations
    data_with_future_dates = pd.concat([raw_data, placeholder_for_future_data])

    # Create lagged rolling features for the future prediction period
    future_lagged = create_lagged_rolling(data_with_future_dates, target_data_column, lags)
    future_features = future_lagged.loc[future_dates].dropna()

    # Ensure feature columns match expected input for the model
    future_features = future_features[[f'{lag}days_lag' for lag in lags]]

    # Predict future stock prices
    future_predictions = model.predict(future_features)

    plot_predictions(y_test, y_predict, future_dates, future_predictions)

make_forecast(model, y_test, y_predict, target_data_column)
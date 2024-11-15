import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# Load and prepare data
close_price = pd.read_excel('abc_close_price.xlsx', index_col='Date', parse_dates=True)
target_raw_data = pd.read_excel('乡村振兴_全国.xlsx', index_col='时间', parse_dates=True)
# hedge_raw_data = pd.read_excel('sse.xlsx', index_col='Date', parse_dates=True)
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

import pandas as pd

def calc_effectiveness(y_test, y_pred, correct_dir_weight=0.5, wrong_dir_weight=1.5):
    # Calculate 5-day lagged returns for both y_test and y_pred
    y_test_5days_yield = (y_test.shift(5) - y_test) * 100 / y_test
    y_pred_5days_yield = (y_pred.shift(5) - y_pred) * 100 / y_pred

    # Combine the results into a single DataFrame and drop NaNs
    df = pd.concat([y_test_5days_yield, y_pred_5days_yield], axis=1, keys=['y_test_5days_yield', 'y_pred_5days_yield']).dropna()

    # Calculate weighted squared errors directly
    def weighted_sqr_error(row):
        same_dir = (row['y_test_5days_yield'] > 0 and row['y_pred_5days_yield'] > 0) or \
                   (row['y_test_5days_yield'] < 0 and row['y_pred_5days_yield'] < 0)
        weight = correct_dir_weight if same_dir else wrong_dir_weight
        return weight * (row['y_test_5days_yield'] - row['y_pred_5days_yield']) ** 2

    # Apply the calculation row-wise
    df['weighted_adjusted_sqr_errors'] = df.apply(weighted_sqr_error, axis=1)

    # Return the sum of weighted squared errors
    return df['weighted_adjusted_sqr_errors'].sum()


def plot_predictions(y_test, y_predict, future_dates=None, future_predictions=None):
    plt.figure(figsize=(12, 6))
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
    y_predict = pd.Series(model.predict(X_test))
    y_predict.index = y_test.index


    # mda = mean_directional_accuracy(y_test, y_predict)
    effectiveness = calc_effectiveness(y_test, y_predict)
    print(f'Effectiveness: {effectiveness:.2f}')

    plot_predictions(y_test, y_predict)
    return model, y_test, y_predict


def make_forecast(model, y_test, y_predict, target_data_column, future_periods=lags[0]):
    # Step 2: Prepare data for future predictions
    # Define future date range
    future_dates = pd.date_range(start=target_raw_data.index[-1] + pd.Timedelta(days=1), periods=future_periods, freq='D')

    # Create a placeholder for future dates with the last known value for simplicity
    placeholder_for_future_data = pd.DataFrame(index=future_dates, columns=target_raw_data.columns)
    placeholder_for_future_data[target_data_column] = target_raw_data[target_data_column].iloc[-1]  # Assuming last value as a proxy

    # Concatenate original and future data for lagged rolling calculations
    data_with_future_dates = pd.concat([target_raw_data, placeholder_for_future_data])

    # Create lagged rolling features for the future prediction period
    future_lagged = create_lagged_rolling(data_with_future_dates, target_data_column, lags)
    future_features = future_lagged.loc[future_dates].dropna()

    # Ensure feature columns match expected input for the model
    future_features = future_features[[f'{lag}days_lag' for lag in lags]]

    # Predict future stock prices
    future_predictions = model.predict(future_features)

    plot_predictions(y_test, y_predict, future_dates, future_predictions)


def mean_directional_accuracy(y_test, y_pred):
    # Calculate the direction of change for both actual and predicted values
    true_direction = np.sign(np.diff(y_test))
    pred_direction = np.sign(np.diff(y_pred))

    # Compare the direction of changes between actual and predicted
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy

model, y_test, y_predict = train_and_backtest_lag(target_raw_data, target_data_column, stock_code, lags)

make_forecast(model, y_test, y_predict, target_data_column)


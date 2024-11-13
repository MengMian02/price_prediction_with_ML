import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load and prepare data
close_price = pd.read_excel('abc_close_price.xlsx', index_col='Date', parse_dates=True)
baidu_index = pd.read_excel('乡村振兴_全国.xlsx', index_col='时间', parse_dates=True)

# Function to create lagged rolling mean columns
def create_lagged_rolling(df, column, lags, window=30):
    lagged_dfs = []
    for lag in lags:
        lagged_df = df[[column]].shift(lag).rolling(window).mean()
        lagged_df.columns = [f'{lag}days_lag']
        lagged_dfs.append(lagged_df)
    return pd.concat(lagged_dfs, axis=1)

# Define lag periods and create lagged columns
lags = [90, 120, 150, 180, 210]
baidu_lagged = create_lagged_rolling(baidu_index, '搜索pc+移动', lags)

# Concatenate DataFrames and drop rows with missing values
df = pd.concat([close_price, baidu_lagged], axis=1).dropna()

# Define features (X) and target (y)
x = df[[f'{lag}days_lag' for lag in lags]]
y = df['1288.HK']  # Convert y to a Series instead of DataFrame for compatibility

# Train/test split and model training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, random_state=0)
model.fit(x_train, y_train)

# Predict and calculate R-squared score
y_predict = model.predict(x_test)
r_square = r2_score(y_test, y_predict)
print(f'R-squared: {r_square:.4f}')

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual Close Price', color='blue')
plt.plot(y_test.index, y_predict, label='Predicted Close Price', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

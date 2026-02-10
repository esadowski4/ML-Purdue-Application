import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load the training data
df = pd.read_csv('tsa_train.csv', parse_dates=['Date'])
df = df.sort_values('Date')

print(f"Data Range: {df.index.min()} to {df.index.max()}")
print(f"Total Days: {len(df)}")
print(df.head())

# Feature engineering
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_of_month'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['day_of_year'] = df['Date'].dt.dayofyear
df['year'] = df['Date'].dt.year
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Data splitting
feature_cols = ['day_of_week', 'day_of_month', 'month', 'day_of_year', 'year', 'is_weekend']
X_train = df.iloc[:-60][feature_cols]
y_train = df.iloc[:-60]['Volume']
X_val = df.iloc[-60:][feature_cols]
y_val = df.iloc[-60:]['Volume']

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
val_prediction = model.predict(X_val)

mae = mean_absolute_error(y_val, val_prediction)

print(f"Average error: {mae:,.0f} passengers")
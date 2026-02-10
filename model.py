import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load the training data
df = pd.read_csv('tsa_train.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# Feature engineering
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_of_month'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['day_of_year'] = df['Date'].dt.dayofyear
df['year'] = df['Date'].dt.year
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

volume_by_date = df.set_index('Date')['Volume'] # Map each date to its corresponding volume

# Map the date 7, 14, 28, and 365 days ago to their corresponding volumes, creating lag features

df['lag7'] = (df['Date'] - pd.Timedelta(days=7)).map(volume_by_date) 
df['lag14'] = (df['Date'] - pd.Timedelta(days=14)).map(volume_by_date)
df['lag28'] = (df['Date'] - pd.Timedelta(days=28)).map(volume_by_date)
df['lag365'] = (df['Date'] - pd.Timedelta(days=365)).map(volume_by_date)
df = df.dropna() # First few rows will have NaN lag values
df = df.reset_index(drop=True) # Reset the index after dropping rows with NaN values

# Data splitting
feature_cols = ['day_of_week', 'day_of_month', 'month', 'day_of_year', 'year', 'is_weekend', 'lag7', 'lag14', 'lag28', 'lag365']
X_train = df.iloc[:-60][feature_cols]
y_train = df.iloc[:-60]['Volume']
X_val = df.iloc[-60:][feature_cols]
y_val = df.iloc[-60:]['Volume']

# Train the model
model = GradientBoostingRegressor(n_estimators=500, # number of trees
    max_depth=4, # maximum depth of each tree (reduces complexity)
    learning_rate=0.05, # slower learning rate means we need more trees, but can lead to better performance
    min_samples_leaf=10, # minimum number of samples required to be at a leaf node (reduces overfitting)
    random_state=42 # for reproducibility
) # empty model

model.fit(X_train, y_train) # show the model the training data
val_prediction = model.predict(X_val) # make predictions on the validation set

mae = mean_absolute_error(y_val, val_prediction)

print(f"\nFeature importances:")
for name, importance in zip(feature_cols, model.feature_importances_):
    print(f"  {name}: {importance:.3f}")

print(f"Average error: {mae:,.0f} passengers")

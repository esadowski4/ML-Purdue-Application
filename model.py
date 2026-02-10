import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
df = pd.read_csv('tsa_train.csv', parse_dates=['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

print(f"Data Range: {df.index.min()} to {df.index.max()}")
print(f"Total Days: {len(df)}")
df.head()
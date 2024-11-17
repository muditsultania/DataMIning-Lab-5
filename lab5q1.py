import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load and clean the dataset
file_path = 'TATAMOTORS_data_2016_2023.csv'
# Step 1: Read and clean the dataset
df = pd.read_csv(file_path, skiprows=2)
df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).reset_index(drop=True)

# Ensure numeric columns are properly formatted
numeric_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Set 'Date' as the index and sort by date
df.set_index('Date', inplace=True)
close_prices = df['Close']

# Step 2: Calculate Moving Averages
df['50_MA'] = close_prices.rolling(window=50).mean()
df['200_MA'] = close_prices.rolling(window=200).mean()
df['365_MA'] = close_prices.rolling(window=365).mean()
df['500_MA'] = close_prices.rolling(window=500).mean()

# Step 3: Plot Close Prices and Moving Averages
plt.figure(figsize=(14, 7))
plt.plot(close_prices, label='Close Prices', color='blue', alpha=0.6)
plt.plot(df['50_MA'], label='50-Day MA', color='orange')
plt.plot(df['200_MA'], label='200-Day MA', color='green')
plt.plot(df['365_MA'], label='365-Day MA', color='red')
plt.plot(df['500_MA'], label='500-Day MA', color='purple')
plt.title('TATA MOTORS Close Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 4: Fit the ARIMA Model
# Differencing to make the data stationary
df['Diff_Close'] = close_prices.diff().dropna()

# Identify ARIMA parameters using ACF and PACF
plot_acf(df['Diff_Close'].dropna(), lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(df['Diff_Close'].dropna(), lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Fit ARIMA model (assume p=1, d=1, q=1 as a starting point)
model = ARIMA(close_prices.dropna(), order=(1, 1, 1))
arima_result = model.fit()

# Step 5: Summarize ARIMA Model
print(arima_result.summary())

# Step 6: Analyze Residuals
residuals = arima_result.resid
plot_acf(residuals, lags=40)
plt.title('Residuals Autocorrelation')
plt.show()

# Save the dataset with moving averages to a new CSV
output_file = 'TATAMOTORS_with_MA.csv'
df.to_csv(output_file)
print(f"Dataset with moving averages saved to: {output_file}")

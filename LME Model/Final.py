#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
date_parser = lambda x: pd.to_datetime(x, format='%Y %B')
data = pd.read_csv(r"C:\Exide\westmetall\output_zinc.csv", parse_dates=['Month'], date_parser = date_parser, index_col='Month')

data.iloc[:, 0:] = data.iloc[:, 0:].replace({',': ''}, regex=True)

# Convert columns after the date column to numeric data types
data.iloc[:, 0:] = data.iloc[:, 0:].apply(pd.to_numeric, errors='coerce')

# Save the modified DataFrame back to a new CSV file
data.to_csv("modified_file.csv", index=True)

# Read the modified CSV file back into a DataFrame
data = pd.read_csv("modified_file.csv", parse_dates=['Month'], index_col='Month')

# Select the first three columns
cols = list(data.columns)[:2]
data = data[cols].astype(float)
data = data.interpolate()
data.to_csv(r"C:\Exide\westmetall\interpolated_zinc_monthly.csv")

target_columns = ['LME Zinc Cash-Settlement', 'LME Zinc 3-month']
target_data = data[target_columns].values

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define a function to create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define the sequence length
seq_length = 10

# Create sequences for training
X, y = create_sequences(data_scaled, seq_length)

# Split the dataset into training and testing sets
split_ratio = 0.67
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the model
model = Sequential([
  Conv1D(filters=3, kernel_size=3, activation="relu", input_shape=(seq_length, 2)),
  LSTM(50, return_sequences=True, activation="relu"),
  Dropout(0.2),  # Add dropout after LSTM layer with rate 0.2
  LSTM(32, activation="relu"),
  Dropout(0.1),  # Add dropout after LSTM layer with rate 0.1
  Dense(2)
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Train the model
model.fit(X_train, y_train, epochs=2000, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

test_dates = data.index[split_index + seq_length:]

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_inv[:, 0], label="True LME Zinc 3-month Price", color="blue")
plt.plot(test_dates, predictions[:, 0], label="Predicted LME Zinc 3-month Price", color="red")
plt.plot(test_dates, y_test_inv[:, 1], label="True LME Zinc Cash-Settlement Price", color="green")
plt.plot(test_dates, predictions[:, 1], label="Predicted LME Zinc Cash-Settlement Price", color="orange")

# Set x-axis labels as dates with desired format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%B'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Adjust locator for desired interval

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("LME Zinc Price Forecasting")
plt.legend()
plt.show()


# In[3]:


predictions


# In[4]:


from sklearn.metrics import mean_absolute_error 

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate MAE
mae = mean_absolute_error(y_test_inv, predictions)
print(f"Mean Absolute Error: {mae}")


# In[26]:


import pandas as pd

# Your prediction array (assuming predictions is a 2D array)

# Define number of days based on the length of predictions array
num_days = len(predictions)



def mean_absolute_percentage_error(y_true, y_pred):
 """Calculates MAPE between true and predicted values.

 Args:
   y_true: Ground truth (actual) values.
   y_pred: Predicted values.

 Returns:
   Mean Absolute Percentage Error.
 """
 return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE for each target variable
mape_lme_3month = 100 - mean_absolute_percentage_error(y_test_inv[:, 0], predictions[:, 0])
mape_lme_cash = 100 - mean_absolute_percentage_error(y_test_inv[:, 1], predictions[:, 1])

# Print MAPE values
print(f"MAPE (LME Zinc 3-month): {mape_lme_3month:.2f}%")
print(f"MAPE (LME Zinc Cash-Settlement): {mape_lme_cash:.2f}%")               



data = pd.DataFrame({
    "Predicted LME Zinc 3-month": predictions[:, 0],
    "Predicted LME Zinc Cash-Settlement": predictions[:, 1],
    "MAPE LME Zinc 3-month": mape_lme_3month,  # Add MAPE values directly
    "MAPE LME Zinc Cash-Settlement": mape_lme_cash
})

data.to_csv(r"C:\Exide\westmetall\LSTM_predictions_with_accuracy.csv", index=False)


# In[27]:


def combine_csv(original_file, prediction_file, output_file):
  """
  Combines two CSV files into separate sheets in an Excel workbook.

  Args:
      original_file (str): Path to the original CSV file.
      prediction_file (str): Path to the prediction CSV file.
      output_file (str): Path to the output Excel workbook.
  """
  # Create full paths for the files incorporating the directory
  original_file = f"C:\\Exide\\westmetall\\{original_file}"
  prediction_file = f"C:\\Exide\\westmetall\\{prediction_file}"
  output_file = f"C:\\Exide\\westmetall\\{output_file}"

  # Read CSV files
  original_data = pd.read_csv(r"C:\Exide\westmetall\interpolated_zinc_monthly.csv")
  prediction_data = pd.read_csv(r"C:\Exide\westmetall\LSTM_predictions_with_accuracy.csv")

  # Create a dictionary with dataframes as values
  data = {
      "Original Data": original_data,
      "Predictions": prediction_data
  }

  # Write data to Excel with sheet names from the dictionary keys
  writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
  for sheet_name, data in data.items():
      data.to_excel(writer, sheet_name=sheet_name, index=False)
  writer.close()

# Example usage
original_file = "original_data.csv"
prediction_file = "predictions_file.csv"
output_file = "combined_data_zinc_monthly.xlsx"

combine_csv(original_file, prediction_file, output_file)

print(f"CSV files combined into '{output_file}'.")


# In[ ]:





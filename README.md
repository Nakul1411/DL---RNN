# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
A financial organization aims to predict future stock prices based on historical market data, as accurate predictions can support better investment decisions. However, stock price data is sequential in nature and depends on past trends, making it difficult to model using traditional methods.

To address this, a Recurrent Neural Network (RNN) model will be developed, as RNNs are designed to handle sequential data by retaining information from previous time steps. The model will be trained using historical closing price data so that it can learn patterns and temporal dependencies in stock price movements.

After training, the model will be used to predict future stock prices based on recent data inputs. The performance of the model will be evaluated by comparing predicted values with actual prices.

The objective is to build a model that can capture time-based patterns effectively and provide accurate predictions for future stock trends.
Google stock prices are given in trainset.csv and testset.csv files. 
## DESIGN STEPS
### Step 1 : Collect Historical Data
Obtain historical stock market data containing the closing prices for a specific company from a financial dataset.

### Step 2 : Preprocess the Data
Clean the dataset, handle missing values, and normalize the closing prices using techniques like Min-Max scaling.

### Step 3 : Create Time-Series Sequences
Convert the data into input sequences where previous n days of closing prices are used to predict the next day's price.

### Step 4 : Build the RNN Model
Design a Recurrent Neural Network architecture with input layer, one or more RNN/LSTM layers, and a dense output layer.

### Step 5 : Train the Model
Feed the training sequences into the RNN and update weights using backpropagation through time to minimize prediction error.

### Step 6 : Predict and Evaluate
Use the trained model to predict future stock prices and evaluate performance using metrics such as Mean Squared Error (MSE).
## PROGRAM

### Name: SURYANARAYANAN T

### Register Number:212224040341

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

df_train.head()
df_test.head()

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  #   Take the output of the last time step
        return out

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from functools import total_ordering
## Step 3: Train the Model
def train_model(model,train_loader,criterian,optimizer,epochs=20):
  model.train()
  train_losses = []
  for epoch in range(epochs):
      total_loss = 0
      epoch_loss = 0
      for x_batch, y_batch in train_loader:
          x_batch, y_batch = x_batch.to(device), y_batch.to(device)
          optimizer.zero_grad()
          outputs = model(x_batch)
          loss = criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
      train_losses.append(total_loss/len(train_loader))
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
  # Plot training loss
  print('Name:SURYANARAYANAN T')
  print('Register Number:212224040341')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()

train_model(model,train_loader,criterion,optimizer,epochs=20)

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:SURYANARAYANAN T')
print('Register Number:212224040341')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')


```

### OUTPUT

<img width="580" height="541" alt="image" src="https://github.com/user-attachments/assets/869d0e92-4e51-44e9-a7ed-32258f959b6f" />

<img width="817" height="623" alt="image" src="https://github.com/user-attachments/assets/a66ccbb8-ed10-4f87-8107-035d5354dac0" />


## Training Loss Over Epochs Plot

## True Stock Price, Predicted Stock Price vs time
<img width="1037" height="657" alt="Screenshot 2025-10-22 165258" src="https://github.com/user-attachments/assets/4ea0bd01-0266-4570-9572-233a8996e4b6" />

### Predictions
<img width="348" height="60" alt="Screenshot 2025-10-22 165306" src="https://github.com/user-attachments/assets/275fceac-102a-458e-af91-c9465deffc54" />

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.

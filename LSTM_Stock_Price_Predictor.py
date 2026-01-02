# Import necessary libraries

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# HAS TO BE GPU TO WORK (IT'S NOT FUNCTIONAL JUST A BASIC CONCEPT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# Pick a stock
ticker = 'GOOGL'

# Download the data of the stock from Jan 1st 2020 to now
df = yf.download(ticker, '2020-01-01')

# Plot the stock's movement based on the data
df.Close.plot(figsize=(12, 8))

# Keep raw prices untouched
closed_raw = df['Close'].values.reshape(-1, 1)

scaler = StandardScaler()
closed_scaled = scaler.fit_transform(closed_raw)

# Number of days it looks at (29 days) to predict the last day (day 30)
seq_length = 30
X, y = [], []


for i in range(len(df) - seq_length):
  # Looping through the data to add every sequence length into a list
  X.append(closed_scaled[i:i+seq_length])
  y.append(closed_scaled[i+seq_length])

# Converting the data list into matrix form using numpy
X = np.array(X)
y = np.array(y)

# Setting 80% of the data as training data
training_set = int(0.8 * len(X))

# Defining the portions of the data that is testing and training
X_train = torch.tensor(X[:training_set], dtype=torch.float32).to(device)
y_train = torch.tensor(y[:training_set], dtype=torch.float32).to(device)

X_test  = torch.tensor(X[training_set:], dtype=torch.float32).to(device)
y_test  = torch.tensor(y[training_set:], dtype=torch.float32).to(device)

# Create our model (basic architecture)
class prediction_model(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    super(prediction_model, self).__init__()

    self.num_layers = num_layers
    self.hidden_dim = hidden_dim

    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

    out, _ = self.lstm(x, (h0, c0))

    # out shape: (batch, seq_len, hidden_dim)
    out = out[:, -1, :]          # take LAST timestep only
    out = self.fc(out)           # (batch, 1)

    return out

# Set model's hyperparameters
model = prediction_model(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)

# Use Mean Squared Error for loss function
loss_function = nn.MSELoss()

# Use Adam optimizer to perform gradient descent
optimizer = optim.Adam(model.parameters(), lr=0.01) # Set learning rate to 0.01

# Set epochs
epochs = 200

for epoch in range(epochs):
  # Set model to training mode
  model.train()

  # Define prediction for y_train
  y_train_pred = model(X_train)

  # Calculate loss
  loss = loss_function(y_train_pred, y_train)

  # Print out epoch numbers and loss during each epoch
  if epoch % 25 == 0:
    print(f'Epoch: {epoch}, Loss: {loss}')

  # Optimizer zero grad
  optimizer.zero_grad()

  # Perform backpropagation on the loss
  loss.backward()

  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  # Step the optimizer (perform gradient descent)
  optimizer.step()


model.eval()

# Last 30 scaled days for inference
last_30_days = closed_scaled[-seq_length:]          # (30, 1)

last_30_tensor = torch.tensor(
    last_30_days,
    dtype=torch.float32
).unsqueeze(0).to(device)                           # (1, 30, 1)

with torch.inference_mode():
    next_day_scaled = model(last_30_tensor)         # (1, 1)

# Inverse scale prediction
next_day_price = float(
    scaler.inverse_transform(next_day_scaled.cpu().numpy())[0, 0]
)

# Last REAL close price (unscaled)
last_close_price = float(closed_raw[-1, 0])

signal = "BUY" if next_day_price > last_close_price else "SELL"

print(f"Last Close Price: ${last_close_price:.2f}")
print(f"Predicted Next Close Price: ${next_day_price:.2f}")
print(f"Signal: {signal}")

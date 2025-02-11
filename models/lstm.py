import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error

###########################################
# Sequence generation
###########################################

def create_sequences(df: pd.DataFrame, columns: list, sequence_length: int = 24, window: int = 24):
    """
    Creates sequences of features and a target value.

    The target is the 'USAGE_KWH' value at a time `window` steps ahead of the sequence.
    """
    # Choose the columns to be included in the sequence.

    if 'USAGE_AT' in columns:
        columns.remove('USAGE_AT')
    data = df[columns].values  # shape: (N, num_features)

    X, y = [], []
    for i in range(len(data) - sequence_length - window):
        x_seq = data[i: i + sequence_length, :]
        # The target is the usage value at the end of the forecast window.
        y_val = data[i + sequence_length + window - 1, -1]  # Last column is 'USAGE_KWH'
        X.append(x_seq)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)

    return X, y


############################################
# Define the LSTM Model
############################################

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Initializes the LSTM model.

        Parameters:
          - input_dim: Number of features in the input.
          - hidden_dim: Number of neurons in the LSTM hidden layer.
          - num_layers: Number of LSTM layers.
          - output_dim: Dimension of the output (typically 1 for regression).
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Define the fully connected output layer.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the model.
        Initializes the hidden and cell states on the same device as the input.
        """
        # Initialize hidden state and cell state with zeros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate the LSTM.
        out, _ = self.lstm(x, (h0, c0))
        # Use the output of the last time step.
        out = self.fc(out[:, -1, :])
        return out


def create_model(
            input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            device: torch.device = torch.device("cpu")
        ):
    """
    Instantiates and returns the LSTM model on the specified device.

    Parameters:
      - input_dim: Number of input features.
      - hidden_dim: Number of LSTM neurons.
      - num_layers: Number of LSTM layers.
      - output_dim: Output dimension (usually 1).
      - device: Device to place the model on (e.g., CPU or GPU).

    Returns:
      - model: The instantiated LSTM model.
    """
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    return model


###########################################
# Train the LSTM Model
###########################################

def train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            num_epochs=300,
            initial_lr=0.01,
            gamma=1.0,
            noise_std=0.01
        ):
    """
    Trains the provided LSTM model using the training data and evaluates on validation data.

    Parameters:
      - model: The LSTM model to be trained.
      - X_train: Training features tensor.
      - y_train: Training target tensor.
      - X_val: Validation features tensor.
      - y_val: Validation target tensor.
      - num_epochs: Number of epochs to train.
      - initial_lr: Initial learning rate.
      - gamma: Decay factor for the learning rate scheduler.
      - noise_std: Standard deviation for white Gaussian noise added to X_train.

    Returns:
      - train_losses: List of training losses for each epoch.
      - val_losses: List of validation losses for each epoch.
      - model: The trained model.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Compute adaptive step size as 20% of total epochs (ensuring at least 1 epoch)
    step_size = max(1, int(0.2 * num_epochs))

    # Create learning rate scheduler to decay the learning rate every step_size epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if gamma < 1:
        print(f"Applying Adaptive LR Scheduler: Step Size = {step_size} epochs, Decay Factor = {gamma}")
    if noise_std != 0:
        print(f"Applying gaussian noise: std = {noise_std}")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Add white Gaussian noise to the training input data.
        noise = torch.randn_like(X_train) * noise_std
        noisy_X_train = X_train + noise

        # Forward pass on noisy training data.
        outputs = model(noisy_X_train)
        train_loss = criterion(outputs, y_train)

        # Backward pass and optimization.
        train_loss.backward()
        optimizer.step()

        # Evaluation on validation set (no gradient computation)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        # Update learning rate.
        scheduler.step()

        # Record losses.
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return train_losses, val_losses, model


###########################################
# Evaluate the LSTM Model
###########################################

def evaluate_model(model, X_val, y_val, scaler_y):
    """
    Evaluates the provided model on the validation set.
    """

    # Ensure the model is in evaluation mode.
    model.eval()
    with torch.no_grad():
        # Forward pass: Get predictions on X_val.
        predictions = model(X_val)

    # Move predictions and y_val to CPU and convert to numpy arrays.
    predictions_np = predictions.cpu().numpy()
    y_val_np = y_val.cpu().numpy()

    # Inverse scale the predictions and true values.
    predictions_inverse = scaler_y.inverse_transform(predictions_np)
    y_val_inverse = scaler_y.inverse_transform(y_val_np)

    # Compute metrics.
    mae = mean_absolute_error(y_val_inverse, predictions_inverse)
    rmse = np.sqrt(mean_squared_error(y_val_inverse, predictions_inverse))

    # Compute MAPE: avoid division by zero by adding a small epsilon.
    epsilon = 1e-10
    mape = np.mean(np.abs((y_val_inverse - predictions_inverse) / (y_val_inverse + epsilon))) * 100.0

    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'RMSE: {rmse:.4f}')

    sum_actual = np.sum(y_val_inverse)
    sum_pred = np.sum(predictions_inverse)
    diff = sum_actual - sum_pred
    print(f"kWh actual: {sum_actual}, kWh pred: {sum_pred}, diff (%): {(abs(diff)/sum_actual)*100}")

    # Plot predicted vs actual.
    plt.figure(figsize=(18, 6))
    plt.plot(y_val_inverse, label='Actual', color='blue')
    plt.plot(predictions_inverse, label='Predicted', color='red', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('USAGE_KWH')
    plt.title('Predicted vs Actual USAGE_KWH')
    plt.legend()
    plt.show()

    # Plot difference between actual and predicted
    plt.figure(figsize=(18, 6))
    plt.plot(y_val_inverse - predictions_inverse, label='Difference', color='red', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('USAGE_KWH')
    plt.title('Difference between Actual and Predicted USAGE_KWH')
    plt.legend()
    plt.show()

    return mae, mape, rmse
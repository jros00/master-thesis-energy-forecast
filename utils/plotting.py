import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

def plot_usage(df: pd.DataFrame):
    """
    Simple plotting function for usage over time.
    """
    plt.figure(figsize=(18, 6))
    plt.plot(df['USAGE_AT'], df['USAGE_KWH'])
    plt.xlabel("Time")
    plt.ylabel("Usage (KWH)")
    plt.title("Usage over Time")
    plt.show()


def plot_train_val_losses(train_losses, val_losses):
    """
        Plots the training and validation losses over epochs.
    """
    plt.figure(figsize=(18, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Training Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_train_val(y_train, y_val):
    """
    Plots training and validation target values side by side.
    
    Parameters:
      - y_train: training target values (numpy array or torch tensor)
      - y_val: validation target values (numpy array or torch tensor)
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().detach().numpy()
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().detach().numpy()
    
    # Remove any extra dimensions
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    
    # Create x-axis indices for train and validation segments
    train_indices = np.arange(len(y_train))
    val_indices = np.arange(len(y_train), len(y_train) + len(y_val))
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_indices, y_train, label="Train", color="blue")
    plt.plot(val_indices, y_val, label="Validation", color="orange")
    plt.xlabel("Time Index")
    plt.ylabel("Target Value")
    plt.title("Train vs. Validation Targets")
    plt.legend()
    plt.show()

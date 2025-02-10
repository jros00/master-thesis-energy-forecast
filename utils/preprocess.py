import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch


def interpolate_missing_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
        Interpolates missing hourly data in 'USAGE_KWH'.
    """
    df['USAGE_AT'] = pd.to_datetime(df['USAGE_AT'], utc=True)
    df = df.set_index('USAGE_AT')
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h', tz=df.index.tz)
    df_full = df.reindex(full_index)
    df_full['USAGE_KWH'] = df_full['USAGE_KWH'].interpolate(method='time')
    df_full = df_full.reset_index().rename(columns={'index': 'USAGE_AT'})

    return df_full


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Adds time-based features to the DataFrame and sorts by time.
    """
    df['USAGE_AT'] = pd.to_datetime(df['USAGE_AT'])
    df['MONTH'] = df['USAGE_AT'].dt.month
    df['YEAR'] = df['USAGE_AT'].dt.year
    df['WEEKDAY'] = df['USAGE_AT'].dt.weekday
    df['DAY_OF_MONTH'] = df['USAGE_AT'].dt.day
    df['HOUR'] = df['USAGE_AT'].dt.hour
    df = df[['USAGE_AT', 'MONTH', 'DAY_OF_MONTH', 'WEEKDAY', 'HOUR', 'USAGE_KWH']]
    df = df.sort_values("USAGE_AT")

    return df


def add_spotprice(df: pd.DataFrame, path_to_spot: str, se_area: str) -> pd.DataFrame:
    """
        Adds spot price data to the DataFrame.
    """
    df_spot = pd.read_csv(path_to_spot)
    df_spot['SPOTPRICE_VALID_AT'] = pd.to_datetime(df_spot['SPOTPRICE_VALID_AT'], utc=True)
    df_spot = df_spot.loc[df_spot['ELECTRICITY_AREA'] == se_area].drop(columns=['ELECTRICITY_AREA'])
    df_merged = pd.merge(
        left=df,
        right=df_spot,
        left_on='USAGE_AT',
        right_on='SPOTPRICE_VALID_AT',
        how='left'
    ).drop(columns=['SPOTPRICE_VALID_AT'])

    return df_merged


def split_train_val(
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = 0.9,
        scaler_type: str = 'minmax',
        device: torch.device = torch.device("cpu")
    ):

    """
    Splits the sequences into training and validation sets and scales the features and targets.

    Parameters:
      - X: 3D numpy array of features with shape (N, sequence_length, num_features)
      - y: 1D or 2D numpy array of targets
      - train_split: Fraction of data to use for training
      - scaler_type: 'minmax' or 'standard' (for MinMaxScaler or StandardScaler)

    Returns:
      - X_train, y_train: Torch tensors for training
      - X_val, y_val: Torch tensors for validation
      - scaler_x: Fitted scaler object for features (used for later inverse scaling)
      - scaler_y: Fitted scaler object for targets (used for later inverse scaling)
    """
    # 1. Split into training and validation
    train_size = int(train_split * len(X))
    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size].reshape(-1, 1)  # ensure 2D shape
    y_val = y[train_size:].reshape(-1, 1)

    # 2. Flatten the 3D arrays into 2D for scaling. From shape (N, seq_len, num_features) to shape (N, num_features)
    N_train, seq_len, num_features = X_train.shape
    N_val = X_val.shape[0]
    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)

    # 3. Choose the scaler for X based on the parameter
    scaler_type = scaler_type.lower()
    if scaler_type == 'minmax':
        scaler_x = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler_x = StandardScaler()
    else:
        raise ValueError("Invalid scaler_type. Please choose 'minmax' or 'standard'.")

    print(f"Applying {scaler_type} scaling.")

    # Fit only on training features to avoid leakage.
    scaler_x.fit(X_train_flat)
    X_train_scaled_flat = scaler_x.transform(X_train_flat)
    X_val_scaled_flat = scaler_x.transform(X_val_flat)

    # Reshape back to 3D arrays.
    X_train_scaled = X_train_scaled_flat.reshape(N_train, seq_len, num_features)
    X_val_scaled = X_val_scaled_flat.reshape(N_val, seq_len, num_features)

    # 4. Now scale the target values.
    if scaler_type == 'minmax':
        scaler_y = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler_y = StandardScaler()

    # Fit on training targets.
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # 5. Convert to PyTorch tensors.
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

    # Print shapes for confirmation
    print("X_train shape:", X_train_tensor.shape)
    print("y_train shape:", y_train_tensor.shape)
    print("X_val shape:", X_val_tensor.shape)
    print("y_val shape:", y_val_tensor.shape)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, scaler_x, scaler_y
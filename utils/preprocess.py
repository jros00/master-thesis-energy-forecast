import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#############################################
### NOTE: Generating the initial dataframe
#############################################


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


def add_spotprice(df: pd.DataFrame, path_to_spot: str, se_area: str = 'SE3') -> pd.DataFrame:
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

    rows_to_fill = len(df_merged.loc[df_merged['SPOTPRICE_SEK_BY_KWH'].isna()])
    if rows_to_fill > 0:
        df_merged['SPOTPRICE_SEK_BY_KWH'] = df_merged['SPOTPRICE_SEK_BY_KWH'].interpolate(method='linear', limit_direction='both')
        print(f"Warning: Interpolated {rows_to_fill} rows for `SPOTPRICE_SEK_BY_KWH`.")

    columns = df_merged.columns.tolist()
    columns.remove('USAGE_KWH')
    columns.append('USAGE_KWH')
    df_merged = df_merged[columns]

    return df_merged


def add_weather_forecast(
        df: pd.DataFrame, 
        path_to_weather_data: str, 
        columns_to_add: list = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'uv_index', 'direct_radiation', 'diffuse_radiation']
    ):

    df_weather = pd.read_csv(path_to_weather_data)
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    columns_to_add.append('time')
    df_weather = df_weather[columns_to_add]
    df_merged = pd.merge(
        left=df,
        right=df_weather,
        left_on='USAGE_AT',
        right_on='time',
        how='left'
    ).drop(columns=['time'])

    for col in columns_to_add:
        if col == 'time':
            continue
        rows_to_fill = len(df_merged.loc[df_merged[col].isna()])
        if rows_to_fill > 0:
            df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')
            print(f"Warning: Interpolated {rows_to_fill} rows for `{col}`.")

    columns = df_merged.columns.tolist()
    columns.remove('USAGE_KWH')
    columns.append('USAGE_KWH')
    df_merged = df_merged[columns]

    return df_merged


def preprocess_data(
       usage_path: str,
       spot_path: str | None,
       weather_path: str | None,
       se_area: str = 'SE3',
       weather_columns: list = ['temperature_2m', 'precipitation', 'direct_radiation', 'uv_index', 'wind_speed_10m']    
    ):

    df_raw = pd.read_csv(usage_path)
    df = interpolate_missing_hours(df_raw) # Interpolate USAGE_KWH for possible missing hours
    df = add_time_based_features(df)

    if spot_path:
        df = add_spotprice(df, spot_path, se_area)
    
    if weather_path:
        df = add_weather_forecast(df, weather_path, weather_columns)

    return df


#############################################
### NOTE: Encoding categorical features (Work in progress)
#############################################

def encode_categoricals(
        df: pd.DataFrame,
        categorical_features: list = ["MONTH", "DAY_OF_MONTH", "WEEKDAY", "HOUR"]
    ):

    if "HOUR" in categorical_features:
        # Hour of day (0 to 23)
        df["hour_sin"] = np.sin(2 * np.pi * df["HOUR"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["HOUR"] / 24)
    if "WEEKDAY" in categorical_features:
        # Weekday (1 to 7)
        df["weekday_sin"] = np.sin(2 * np.pi * (df["WEEKDAY"] - 1) / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * (df["WEEKDAY"] - 1) / 7)
    if "MONTH" in categorical_features:
        # Month (1 to 12)
        df["month_sin"] = np.sin(2 * np.pi * (df["MONTH"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["MONTH"] - 1) / 12)
    if "DAY_OF_MONTH" in categorical_features:
        # Day of month (1 to 31)
        df["day_sin"] = np.sin(2 * np.pi * (df["DAY_OF_MONTH"] - 1) / 31)
        df["day_cos"] = np.cos(2 * np.pi * (df["DAY_OF_MONTH"] - 1) / 31)

    df = df.drop(columns=categorical_features)
    columns: list = df.columns.values.tolist()
    columns.remove('USAGE_KWH')
    columns.append('USAGE_KWH')
    
    return df[columns]


#############################################
### NOTE: Splitting into X and y (XGBoost)
#############################################


def create_features_and_targets(
        df: pd.DataFrame, 
        forecast_horizon: int = 24, 
        lags: list = None,
        return_feature_names: bool = False
    ):
    """
    Returns features X and targets y such that the target for each row is the USAGE_KWH value
    forecast_horizon hours into the future, and includes lagged features.

    Parameters:
      - df: DataFrame with at least 'USAGE_AT' and 'USAGE_KWH'.
      - forecast_horizon: How many hours ahead to predict.
      - lags: List of lag values to create as features.
    """
    df = df.copy().sort_values(by="USAGE_AT")
    
    # Create lagged features if specified
    if lags is not None:
        for lag in lags:
            df[f'lag_{lag}'] = df['USAGE_KWH'].shift(lag)
    
    # Shift the target column upward so that the target is forecast_horizon hours ahead
    df['target'] = df['USAGE_KWH'].shift(-forecast_horizon)
    
    # Drop rows where any feature or the target is NaN
    df = df.dropna()
    
    # For features, drop the original 'USAGE_AT' and 'USAGE_KWH' columns if they're not needed
    X_df = df.drop(columns=['USAGE_AT', 'USAGE_KWH', 'target'])
    feature_names = X_df.columns.tolist()
    X = X_df.values
    y = df['target'].values
    
    if return_feature_names:
        return X, y, feature_names
    return X, y


#############################################
### NOTE: Splitting into X and y (LSTM)
#############################################

def create_sequences(
        df: pd.DataFrame, 
        columns: list, 
        sequence_length: int = 24, 
        window: int = 24, 
        return_feature_names: bool = False
    ):
    """
    Creates sequences of features and a target value.

    The target is the 'USAGE_KWH' value at a time `window` steps ahead of the sequence.
    """
    # Choose the columns to be included in the sequence.

    if 'USAGE_KWH' in columns:
        columns.remove('USAGE_KWH')
        columns.append('USAGE_KWH')
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

    if return_feature_names:
        feature_names = columns
        return X, y, feature_names
    return X, y


#############################################
### NOTE: Splitting into train and val
#############################################

def split_train_val(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        train_split: float = 0.9,
        scaler_type: str = 'minmax',
        scale_columns: list = ["USAGE_KWH"],
        device: torch.device | None = torch.device("cpu")
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

    # Identify which columns to scale by index
    scale_idxs = [feature_names.index(col) for col in scale_columns]

    # Split into training and validation
    train_size = int(train_split * len(X))
    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size].reshape(-1, 1)  # ensure 2D shape
    y_val = y[train_size:].reshape(-1, 1)

    # If needed, flatten the 3D arrays into 2D for scaling. From shape (N, seq_len, num_features) to shape (N, num_features)
    if len(X_train.shape) == 3:
        N_train, seq_len, num_features = X_train.shape
    elif len(X_train.shape) == 2:
        N_train, num_features = X_train.shape

    N_val = X_val.shape[0]
    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)

    # Choose the scaler for X based on the parameter
    scaler_type = scaler_type.lower()
    if scaler_type == 'minmax':
        scaler_x = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler_x = StandardScaler()
    else:
        raise ValueError("Invalid scaler_type. Please choose 'minmax' or 'standard'.")

    print(f"Applying {scaler_type} scaling.")

    # Fit only on the columns we want to scale in the training set and only on training features to avoid leakage.
    scaler_x.fit(X_train_flat[:, scale_idxs])

    # 3) Transform only those columns, leave the others unchanged
    X_train_scaled_flat = X_train_flat.copy()
    X_val_scaled_flat = X_val_flat.copy()

    X_train_scaled_flat[:, scale_idxs] = scaler_x.transform(X_train_flat[:, scale_idxs])
    X_val_scaled_flat[:, scale_idxs] = scaler_x.transform(X_val_flat[:, scale_idxs])

    # If needed, reshape back to 3D arrays.
    if len(X_train.shape) == 3:
        X_train_scaled = X_train_scaled_flat.reshape(N_train, seq_len, num_features)
        X_val_scaled = X_val_scaled_flat.reshape(N_val, seq_len, num_features)
    elif len(X_train.shape) == 2:
        X_train_scaled = X_train_scaled_flat.copy()
        X_val_scaled = X_val_scaled_flat.copy()

    # 4. Now scale the target values.
    if scaler_type == 'minmax':
        scaler_y = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler_y = StandardScaler()

    # Fit on training targets.
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device) if device is not None else X_train_scaled
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device) if device is not None else X_val_scaled
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device) if device is not None else y_train_scaled
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device) if device is not None else y_val_scaled

    # Print shapes for confirmation
    print("X_train shape:", X_train_tensor.shape)
    print("y_train shape:", y_train_tensor.shape)
    print("X_val shape:", X_val_tensor.shape)
    print("y_val shape:", y_val_tensor.shape)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, scaler_x, scaler_y


if __name__ == '__main__':
    df = pd.read_csv('data/cons_sthlm_mean.csv')
    print(len(df))
    df = interpolate_missing_hours(df)
    print(len(df))
    df = add_time_based_features(df)
    print(len(df))
    df = add_spotprice(df, 'data/spotprices.csv', se_area='SE3')
    print(len(df))
    df = add_weather_forecast(df, 'data/open-meteo-stockholm.csv', columns_to_add=['temperature_2m'])
    print(len(df))
    print(f"\n{df}")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def get_autocorrelation_lags(
        df: pd.DataFrame, 
        N=3
    ):
    """ 
      Returns the top N autocorrelation lags, can be used as input to the `create_lagged_features` method. 
    """

    df = df.sort_values(by="USAGE_AT")
    lags = np.arange(0, 8760)
    autocorr_values = [[lag, df['USAGE_KWH'].autocorr(lag)] for lag in lags]

    # Sort by highest autocorrelation and filter out lags < 24
    autocorr_values = sorted(
        [[int(x[0]), float(x[1])] for x in autocorr_values if x[0] >= 24],  # Filter condition
        key=lambda x: x[1],  # Sort by autocorrelation value
        reverse=True  # Highest correlation first
    )
    autocorr_values = autocorr_values[:N]

    print(f"Top {N} lags:")
    print(autocorr_values)

    autocorr_indexes = [int(x[0]) for x in autocorr_values]

    return autocorr_indexes


def create_features_and_targets(df: pd.DataFrame, forecast_horizon: int = 24, lags: list = None):
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
    X = df.drop(columns=['USAGE_AT', 'USAGE_KWH', 'target']).values
    y = df['target'].values
    
    return X, y


def create_model(
            n_estimators: int,
            learning_rate: float,
            random_state: int
        ):
    """
    Instantiates and returns the XGBoost model on the specified device.

    Parameters:
      - n_estimators: Number of weak learners.
      - learning_rate
      - random_state

    Returns:
      - model: The instantiated XGBoost model.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    return model


def train_model(
        model: XGBRegressor, 
        X_train, 
        y_train
    ):

    model.fit(X_train, y_train)
    return model


def evaluate_model(
        model: XGBRegressor, 
        X_val, 
        y_val, 
        scaler_y
    ):
    
    """
    Evaluates the provided XGBoost model on the validation set.
    """

    predictions = np.array(model.predict(X_val)).reshape(-1, 1)

    # Inverse scale the predictions and true values.
    predictions_inverse = scaler_y.inverse_transform(predictions)
    y_val_inverse = scaler_y.inverse_transform(y_val)

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_train_val_dfs(
        df: pd.DataFrame,
        train_frac: int
    ):

    df = df.copy()

    df['USAGE_AT'] = pd.to_datetime(df['USAGE_AT'])
    df = df.sort_values('USAGE_AT')

    # Create a numerical time index (assuming hourly data)
    df['time_idx'] = (df['USAGE_AT'] - df['USAGE_AT'].min()).dt.total_seconds() // 3600
    df['time_idx'] = df['time_idx'].astype(int)

    # Create a dummy group id (useful if you only have one time series)
    df['group'] = 1

    columns: list = df.columns.values.tolist()
    columns.remove('USAGE_KWH')
    columns.append('USAGE_KWH')
    df = df[columns]

    max_time = df["time_idx"].max()
    train_cutoff = int(train_frac * max_time)

    df_train = df[df.time_idx <= train_cutoff]
    df_val = df[df.time_idx > train_cutoff]

    return df_train, df_val


def get_train_val_datasets(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        max_prediction_length: int = 24,
        max_encoder_length: int = 168,
        time_varying_known_categoricals: list | None = None,
        time_varying_known_reals: list = [
            "time_idx", 
            "MONTH", "DAY_OF_MONTH", "WEEKDAY", "HOUR",
            "temperature_2m", "precipitation", "direct_radiation", 
            "uv_index", "wind_speed_10m"
        ],
        scalers : dict | None = None
    ):

    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="USAGE_KWH",
        group_ids=["group"],
        min_encoder_length=max_encoder_length,  # history length (e.g., 168 hours)
        max_encoder_length=max_encoder_length,  # fixed history length
        min_prediction_length=max_prediction_length,  # forecast horizon (24 hours)
        max_prediction_length=max_prediction_length,  # forecast horizon (24 hours)
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        scalers=scalers,
        time_varying_unknown_reals=["USAGE_KWH"],
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus"
        ),
    )

    # NOTE: re‐uses the training set’s encoders/scalers for normalization instead of fitting them again. 
    # This helps avoid data leakage from validation.
    validation = TimeSeriesDataSet.from_dataset(
        training, df_val, predict=False, stop_randomization=True
    )

    print(f"Training samples: {len(training)}")
    print(f"Validation samples: {len(validation)}")

    return training, validation


def get_train_val_loaders(
        training: TimeSeriesDataSet,
        validation: TimeSeriesDataSet,
        batch_size: int = 64,
        num_workers: int = 0
    ):
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


def create_model(
        training: TimeSeriesDataSet,
        lr: float = 0.03,
        hidden_size: int = 16,       
        attention_head_size: int = 1,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        output_size: int = 7,
        loss = QuantileLoss(),
        log_interval: int = 10,
        reduce_on_plateau_patience: int = 4
    ):

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=hidden_size,                            # Size of hidden layers (model capacity)
        attention_head_size=attention_head_size,            # Number of attention heads
        dropout=dropout,                                    # Dropout rate for regularization
        hidden_continuous_size=hidden_continuous_size,      # Hidden size for continuous variables
        output_size=output_size,                            # Number of quantile outputs (for probabilistic forecasting)
        loss=loss,                                          # Use QuantileLoss for probabilistic forecasting
        log_interval=log_interval,
        reduce_on_plateau_patience=reduce_on_plateau_patience,
    )

    return tft


def train_model(
        model: TemporalFusionTransformer,
        train_loader,
        val_loader,
        max_epochs: int = 10,
        devices = 1
    ):

    logger = CSVLogger("logs", name="tft")
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        logger=logger
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    trained_model = model
    return trained_model



def evaluate_model(
        model: TemporalFusionTransformer, 
        dataloader
    ):

    model.eval()
    with torch.no_grad():
        predictions = model.predict(dataloader)

    # Ensure predictions is a NumPy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # Collect actual target values from each batch.
    actuals = []
    for batch in dataloader:
        # Batch is a tuple: (input_dict, (target, None))
        # Extract the target tensor and move it to CPU before converting to NumPy.
        target = batch[1][0]
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        actuals.append(target)
    actuals = np.concatenate(actuals, axis=0)

    predictions_flat = []
    for i, x in enumerate(predictions):
        if i % 24 == 0: # Only keep the forecasts of the next 24 hours, dischard 1-23
            for el in x:
                predictions_flat.append(el)


    actuals_flat = []
    for i, x in enumerate(actuals):
        if i % 24 == 0: # Only keep the forecasts of the next 24 hours, dischard 1-23
            for el in x:
                actuals_flat.append(el)


    predictions_flat = np.array(predictions_flat.copy())
    actuals_flat = np.array(actuals_flat.copy())

    mae = mean_absolute_error(actuals_flat, predictions_flat)
    rmse = np.sqrt(mean_squared_error(actuals_flat, predictions_flat))
    epsilon = 1e-10
    mape = np.mean(np.abs((actuals_flat - predictions_flat) / (actuals_flat + epsilon))) * 100.0

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    sum_actual = np.sum(actuals_flat)
    sum_pred = np.sum(predictions_flat)
    diff = sum_actual - sum_pred
    print(f"kWh actual: {sum_actual}, kWh pred: {sum_pred}, diff (%): {(abs(diff)/sum_actual)*100}")

    # Optionally, plot the predictions vs. actuals.
    plt.figure(figsize=(18, 6))
    plt.plot(actuals_flat, label="Actual", color="blue")
    plt.plot(predictions_flat, label="Predicted", color="red", alpha=0.7)
    plt.xlabel("Sample")
    plt.ylabel("USAGE_KWH")
    plt.title("Predicted vs Actual USAGE_KWH")
    plt.legend()
    plt.show()

    return predictions_flat

# Electricity Forecasting for Enhanced Day-Ahead Market Bidding

This repository contains the code for my master's thesis project on forecasting electricity consumption and production to enhance bidding strategies in the Nordpool day-ahead market. The project compares traditional time series models (e.g., ARIMA, SARIMA) with advanced deep learning methods (LSTM) to improve forecast accuracy and inform market bidding decisions.

### Project status: IN PROGRESS (NOT COMPLETED)

### Repository Structure:

Repository Structure:
---------------------

```
.
+-- models/
|     +-- lstm.py         (LSTM model definition and related code)
|     +-- other_models.py (Optional: Additional model implementations)
+-- utils/
|     +-- preprocessing.py (Data cleaning, interpolation, and feature engineering)
|     +-- plotting.py      (Plotting functions for usage, forecasts, and evaluation)
+-- data/               (Optional: Data files or scripts to download datasets)
+-- experiments/        (Optional: Experiment scripts and notebooks)
+-- model.ipynb         (The different models, evaluated)
+-- readme.md           (This file)
```

### Requirements:
- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PyTorch
- etc ...

Installation:
Install dependencies via:
    pip install -r requirements.txt
(Ensure that your requirements.txt includes the above packages.)

Usage:

1. Data Preprocessing:
   Use the functions in utils/preprocessing.py to clean your data, interpolate missing values, and add time-based features.
   Example:
       from utils.preprocessing import interpolate_missing_hours, add_time_based_features
       df = interpolate_missing_hours(raw_df)
       df = add_time_based_features(df)

2. Model Training:
   Model-specific code is located under models/.
   For example, to train the LSTM model:
       from models.lstm_model import LSTMModel, create_model, train_model
       # Prepare your data tensors (X_train, y_train, X_val, y_val)
       model = create_model(input_dim, hidden_dim, num_layers, output_dim, device)
       train_losses, val_losses, trained_model = train_model(model, X_train, y_train, X_val, y_val, num_epochs=300)

3. Evaluation & Plotting:
   Evaluate the trained model and plot predicted vs. actual values using the functions in utils/plotting.py.
   Example:
       from utils.plotting import evaluate_model
       mae, mape, rmse = evaluate_model(trained_model, X_val, y_val, scaler_y)

Project Overview:
- Objective: Improve forecasting accuracy of electricity consumption and production to optimize day-ahead market bidding at Nordpool.
- Approach: Compare traditional models (ARIMA/SARIMA) with LSTM-based deep learning methods.
- Evaluation: Forecast performance is assessed using MAE, MAPE, and RMSE. Visualizations include predicted vs. actual plots with inverse-scaled outputs.

License:
MIT License (See LICENSE file for details)

Contact:
For questions or feedback, please reach out at johannes@rosing.se.

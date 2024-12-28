# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Load Data
    logging.info("Loading data...")
    data = pd.read_csv("data/processed/duration_data.csv")
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    data.set_index('Date', inplace=True)
except FileNotFoundError:
    logging.error("Data file not found. Ensure the file path is correct.")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred while loading the data: {e}")
    raise

# Prepare Data for Feature Importance Analysis
try:
    logging.info("Preparing predictors and target variable...")
    predictors = ['temperature', 'humidity', 'precipitation', 'apparent temperature', 'pressure', 'cloud_cover', 'wind_speed', 'Public Holiday', 'Working Day', 'Weekday']
    y = data['Average Duration']
except KeyError as e:
    logging.error(f"Missing expected column in data: {e}")
    raise

# Split Ratios and Normalization Options
splits = {'70:30': 0.7, '60:40': 0.6, '80:20': 0.8}
scalers = {'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}

# Initialize Results DataFrame
performance_results = []

for split_name, split_ratio in splits.items():
    for norm_name, scaler in scalers.items():
        try:
            # Train-Test Split
            logging.info(f"Processing split {split_name} with normalization {norm_name}...")
            train_size = int(len(data) * split_ratio)
            train, test = data.iloc[:train_size], data.iloc[train_size:]

            X_train, y_train = train[predictors], train['Average Duration']
            X_test, y_test = test[predictors], test['Average Duration']

            # Normalize Data
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model 1: Random Forest Regressor
            rf_params = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20]
            }
            rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, scoring='neg_mean_squared_error', cv=3, verbose=0)
            rf.fit(X_train_scaled, y_train)
            y_pred_rf = rf.best_estimator_.predict(X_test_scaled)
            rf_performance = {
                'Model': 'Random Forest',
                'Split': split_name,
                'Normalization': norm_name,
                'MAE': mean_absolute_error(y_test, y_pred_rf),
                'RMSE': mean_squared_error(y_test, y_pred_rf, squared=False),
                'R2': r2_score(y_test, y_pred_rf)
            }
            performance_results.append(rf_performance)

            # Save Predictions
            rf_predictions = test.copy()
            rf_predictions['RF_Predictions'] = y_pred_rf
            rf_predictions.to_csv(f'rf_predictions_{split_name}_{norm_name}.csv', index=True)

            # Model 2: Gradient Boosting Regressor
            gb_params = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            gb = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, scoring='neg_mean_squared_error', cv=3, verbose=0)
            gb.fit(X_train_scaled, y_train)
            y_pred_gb = gb.best_estimator_.predict(X_test_scaled)
            gb_performance = {
                'Model': 'Gradient Boosting',
                'Split': split_name,
                'Normalization': norm_name,
                'MAE': mean_absolute_error(y_test, y_pred_gb),
                'RMSE': mean_squared_error(y_test, y_pred_gb, squared=False),
                'R2': r2_score(y_test, y_pred_gb)
            }
            performance_results.append(gb_performance)

            # Save Predictions
            gb_predictions = test.copy()
            gb_predictions['GB_Predictions'] = y_pred_gb
            gb_predictions.to_csv(f'gb_predictions_{split_name}_{norm_name}.csv', index=True)

            # Model 3: Neural Network
            nn = Sequential([
                Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1)
            ])
            nn.compile(optimizer='adam', loss='mse')
            nn.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)
            y_pred_nn = nn.predict(X_test_scaled).flatten()
            nn_performance = {
                'Model': 'Neural Network',
                'Split': split_name,
                'Normalization': norm_name,
                'MAE': mean_absolute_error(y_test, y_pred_nn),
                'RMSE': mean_squared_error(y_test, y_pred_nn, squared=False),
                'R2': r2_score(y_test, y_pred_nn)
            }
            performance_results.append(nn_performance)

            # Save Predictions
            nn_predictions = test.copy()
            nn_predictions['NN_Predictions'] = y_pred_nn
            nn_predictions.to_csv(f'nn_predictions_{split_name}_{norm_name}.csv', index=True)

        except Exception as e:
            logging.error(f"Error during processing for split {split_name}, normalization {norm_name}: {e}")

# Save Performance Results
try:
    performance_df = pd.DataFrame(performance_results)
    performance_df.to_csv('model_performance_results.csv', index=False)
    logging.info("Model performance results saved successfully.")
except Exception as e:
    logging.error(f"Error saving performance results: {e}")
    raise

# Print Summary
logging.info("Model Performance Summary:")
logging.info(performance_df)

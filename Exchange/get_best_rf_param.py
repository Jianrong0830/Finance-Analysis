# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the data (replace 'data.csv' with your actual file path)
data_path = 'data/data.csv'
data = pd.read_csv(data_path)

# Check for missing values
missing_values = data.isnull().sum()

# Define predictors and target variable
X = data.drop(['下一交易日USDTWD收盤價', 'Unnamed: 0'], axis=1)
y = data['下一交易日USDTWD收盤價']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)
mlp_model = MLPRegressor(random_state=42, max_iter=500)

# Train the models
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
mlp_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test)

# Evaluate the models using RMSE
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))

# Feature importance from Random Forest
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)

# Plotting Feature Importance
plt.figure(figsize=(10, 8))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance - Random Forest')
plt.show()

# Parameter grid for full GridSearchCV
param_grid_rf_full = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with full parameter grid
grid_search_rf_full = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf_full, 
                                   cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the model using GridSearchCV
grid_search_rf_full.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
best_params_rf_grid = grid_search_rf_full.best_params_

print(best_params_rf_grid)

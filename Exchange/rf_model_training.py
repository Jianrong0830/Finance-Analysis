# Re-importing necessary libraries and loading the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the data
data_path = 'data/data.csv'
data = pd.read_csv(data_path)

# Define predictors and target variable
X = data.drop(['下一交易日USDTWD收盤價', 'Unnamed: 0'], axis=1)
y = data['下一交易日USDTWD收盤價']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model with the best parameters
best_rf_model = RandomForestRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, 
                                      n_estimators=50, random_state=30)
best_rf_model.fit(X_train, y_train)

joblib.dump(best_rf_model, 'model/rf_model_1.joblib')
joblib.dump(scaler, 'model/scaler_1.joblib')

# Make predictions on the test set
y_pred_best_rf = best_rf_model.predict(X_test)

# Evaluate the model using RMSE
rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
print(rmse_best_rf)

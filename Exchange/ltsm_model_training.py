# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")


# Reload the data to check its structure
data_path = 'data/data.csv'
data = pd.read_csv(data_path)

# Drop the non-numeric "Unnamed: 0" column and isolate target variable
X = data.drop(['下一交易日USDTWD收盤價', 'Unnamed: 0', '時間序列'], axis=1)
y = data['下一交易日USDTWD收盤價']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM model
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Initialize the LSTM model
model = Sequential()

# Add layers
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(LSTM(units=100, return_sequences=True, kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(LSTM(units=100, return_sequences=True, kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(LSTM(units=100, kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=8)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Save the model
if(r2>0):
    model.save('model/ltsm_model1.h5')
    print('better')

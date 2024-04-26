# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('data/data.csv')

# Define predictors and target variable
X = data.drop(['下一交易日USDTWD收盤價', 'Unnamed: 0'], axis=1)
y = data['下一交易日USDTWD收盤價']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create an advanced neural network model
advanced_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(1)
])

# Compile the model with Adam optimizer and learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

advanced_model.compile(optimizer=optimizer, loss='mse')

# Train the model
advanced_model.fit(X_train, y_train, epochs=1000, batch_size=5, validation_split=0.1, callbacks=[early_stopping])

# Make predictions
y_pred = advanced_model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Save the model
if(r2>0.9279317434764545):
    advanced_model.save('model/mlp_model1.h5')
    print('better')

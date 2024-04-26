from joblib import load
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def predict(data_series):
    model = load('model/rf_model_1.joblib')
    scaler = load('model/scaler_1.joblib')
    features = data_series.values.reshape(1, -1)
    scaled = scaler.transform(features)
    predicted_value = model.predict(scaled)
    return predicted_value[0]


data = pd.read_csv('data/data.csv')
data_series = data.drop(['下一交易日USDTWD收盤價', 'Unnamed: 0'], axis=1).iloc[0]
print(predict(data_series))
print(data['下一交易日USDTWD收盤價'].iloc[2000])
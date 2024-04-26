import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 載入數據
df = pd.read_csv('data/raw_data.csv')

# 將數據轉換為時間序列格式
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

# 列出所有的解釋變量
exog_variables = df.drop(columns=['標普500'])

# 建立和訓練模型，參數根據實際數據調整
model = SARIMAX(df['標普500'], exog=exog_variables, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# 進行預測
pred = model_fit.predict(start=pd.to_datetime('2023-04-01'), end=pd.to_datetime('2023-05-01'), exog=exog_variables.tail(12), dynamic=False)
print(pred)

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['標普500'], label='標普500')
plt.plot(pred.index, pred, label='Forecast')
plt.legend(loc='best')
plt.title('Historical Data and Forecast')
plt.show()

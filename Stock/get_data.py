import pandas as pd
import yfinance as yf
import quandl
from pandas_datareader import data as pdr

# Quandl的API
quandl.ApiConfig.api_key = 'sg5yH1WFbv6mAekYVj-Z'

# 日期範圍
start_date = '2000-01-01'
end_date = '2030-01-01'

# 大盤指數
indexes=['^GSPC' ,'ES=F']
indexes_data = yf.download(indexes, start=start_date, end=end_date, interval='1d', progress=False)['Close']
indexes_data = pd.DataFrame(indexes_data)
indexes_data = indexes_data.rename(columns={'^GSPC':'標普500', 'ES=F':'標普500期貨'})
indexes_data = indexes_data.reindex(columns=['標普500', '標普500期貨'])

# 大宗商品
commodities = ['CL=F','BZ=F','HO=F','NG=F','GC=F','PL=F','SI=F','HG=F','PA=F','ZC=F','ZW=F','ZR=F','ZS=F','HE=F','GF=F','LE=F','CC=F','KC=F','CT=F','SB=F','LBS=F']
commodity_data = yf.download(commodities, start=start_date, end=end_date, interval='1d', progress=False)['Close']
commodity_data = commodity_data.rename(columns={
    'CL=F': '紐約原油近月', 'BZ=F':'布倫特原油近月', 'HO=F': '熱燃油近月', 'NG=F': '天然氣近月', 'GC=F': '黃金近月', 'PL=F': '白金近月',
    'SI=F': '白銀近月', 'HG=F': '高級銅近月', 'PA=F': '鈀金近月', 'ZC=F': '玉米近月', 'ZW=F': '小麥近月', 'ZR=F': '稻近月',
    'ZS=F': '黃豆近月', 'HE=F': '瘦肉豬近月', 'GF=F': '飼養牛近月', 'LE=F': '活牛近月', 'CC=F': '可可豆近月',
    'KC=F': '咖啡近月', 'CT=F': '棉花近月', 'SB=F': '11號糖近月', 'LBS=F': '隨機長度木材近月'
})
new_columns_order = ['紐約原油近月', '布倫特原油近月', '熱燃油近月', '天然氣近月', '黃金近月', '白金近月',
                     '白銀近月', '高級銅近月', '鈀金近月', '玉米近月', '小麥近月', '稻近月',
                     '黃豆近月', '瘦肉豬近月', '飼養牛近月', '活牛近月', '可可豆近月',
                     '咖啡近月', '棉花近月', '11號糖近月', '隨機長度木材近月']
commodity_data = commodity_data.reindex(columns=new_columns_order)

# 基準利率
interest_rate_data = pdr.get_data_fred('DFF', start_date, end_date)
interest_rate_data = pd.DataFrame(interest_rate_data)
interest_rate_data = interest_rate_data.rename(columns={'DFF':'聯邦資金有效利率'})

# 美元指數
exchange_rate_data = yf.download('DX-Y.NYB', start=start_date, end=end_date, interval='1d', progress=False)['Close']
exchange_rate_data = pd.DataFrame(exchange_rate_data)
exchange_rate_data = exchange_rate_data.rename(columns={'Close':'美元指數'})

# 獲取國債殖利率數據(插植)
bond_yield_data = pdr.get_data_fred(['GS1', 'GS10', 'GS30'], start_date, end_date)
bond_yield_data = bond_yield_data.rename(columns={'GS1':'1年期國債殖利率', 'GS10':'10年期國債殖利率', 'GS30':'30年期國債殖利率'})
bond_yield_data = bond_yield_data.resample('D').interpolate()
bond_yield_data['10/1年期國債殖利率差']=bond_yield_data['10年期國債殖利率']-bond_yield_data['1年期國債殖利率']
bond_yield_data['30/1年期國債殖利率差']=bond_yield_data['30年期國債殖利率']-bond_yield_data['1年期國債殖利率']

# 獲取CPI和PPI數據(插植)
cpi_ppi_data = pdr.get_data_fred(['CPIAUCSL', 'PPIACO'], start_date, end_date)
cpi_ppi_data = cpi_ppi_data.rename(columns={'CPIAUCSL':'CPI', 'PPIACO':'PPI'})
cpi_ppi_data = cpi_ppi_data.resample('D').interpolate()

# 獲取非農就業人數、失業率和GDP數據(插植)
employment_gdp_data = pdr.get_data_fred(['PAYEMS', 'UNRATE', 'GDP'], start_date, end_date)
employment_gdp_data = employment_gdp_data.rename(columns={'PAYEMS':'非農業就業人數', 'UNRATE':'失業率'})
employment_gdp_data = employment_gdp_data.resample('D').interpolate()

# 合併資料
data = pd.merge(indexes_data, commodity_data, left_index=True, right_index=True, how='outer')
data = pd.merge(data, interest_rate_data, left_index=True, right_index=True, how='outer')
data = pd.merge(data, exchange_rate_data, left_index=True, right_index=True, how='outer')
data = pd.merge(data, bond_yield_data, left_index=True, right_index=True, how='outer')
data = pd.merge(data, cpi_ppi_data, left_index=True, right_index=True, how='outer')
data = pd.merge(data, employment_gdp_data, left_index=True, right_index=True, how='outer')

# 時間序列
data['時間序列']=range(len(data))

# 清理空值
data = data.dropna(axis=0)

# 匯出及輸出
data.to_csv('data/data.csv',encoding='utf_8_sig')

'''
variables_to_keep=['標普500','紐約原油近月','天然氣近月','白金近月','白銀近月','高級銅近月','瘦肉豬近月','活牛近月','可可豆近月','咖啡近月','11號糖近月','聯邦資金有效利率','美元指數','30年期國債殖利率','非農業就業人數','時間序列']
filtered=data[variables_to_keep]
filtered.to_csv('data/filtered.csv',encoding='utf_8_sig')
'''

print(data)
print(len(data))


import pandas as pd
import yfinance as yf
import quandl
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings("ignore")

# Quandl的API
quandl.ApiConfig.api_key = 'sg5yH1WFbv6mAekYVj-Z'

# 日期範圍
start_date = '2005-01-01'
end_date = '2023-09-20'

# 美元兌台幣匯率 ---------------------------------------------------------------
usdtwd = yf.download('USDTWD=X', start=start_date, end=end_date, interval='1d', progress=False)[['Close', 'Open', 'High', 'Low']]
usdtwd.rename(columns={'Close': '下一交易日USDTWD收盤價', 'Open': 'USDTWD開盤價', 'High': 'USDTWD最高價', 'Low': 'USDTWD最低價'}, inplace=True)
usdtwd['下一交易日USDTWD收盤價'] = usdtwd['下一交易日USDTWD收盤價'].shift(-1)
usdtwd['下一交易日間隔'] = (usdtwd.index.to_series().shift(-1) - usdtwd.index.to_series()).dt.days

# 美國 ------------------------------------------------------------------------

# 指數(標普500)
gspc = yf.download('^GSPC', start=start_date, end=end_date, interval='1d', progress=False)['Close']
gspc.rename('標普500', inplace=True)
# 國債殖利率(插植)
usa_bond_yield = pdr.get_data_fred(['DGS1MO','DGS3MO','DGS6MO','GS1', 'GS10', 'GS30'], start_date, end_date)
usa_bond_yield = usa_bond_yield.rename(columns={'DGS1MO':'美國近1月國債殖利率', 'DGS3MO':'美國近3月國債殖利率', 'DGS6MO':'美國近6月國債殖利率', 'GS1':'美國1年期國債殖利率', 'GS10':'美國10年期國債殖利率', 'GS30':'美國30年期國債殖利率'})
usa_bond_yield = usa_bond_yield.resample('D').interpolate()
# 基準利率
usa_interest_rate = pdr.get_data_fred('DFF', start_date, end_date)
usa_interest_rate = pd.DataFrame(usa_interest_rate)
usa_interest_rate = usa_interest_rate.rename(columns={'DFF':'美國聯邦資金有效利率'})
# CPI、PPI
usa_cpi_ppi = pdr.get_data_fred(['CPIAUCSL', 'PPIACO'], start_date, end_date)
usa_cpi_ppi = usa_cpi_ppi.rename(columns={'CPIAUCSL':'美國CPI', 'PPIACO':'美國PPI'})
usa_cpi_ppi = usa_cpi_ppi.resample('D').interpolate()
# GDP、失業率、非農就業人數(插植)
usa_employment_gdp = pdr.get_data_fred(['PAYEMS', 'UNRATE', 'GDP'], start_date, end_date)
usa_employment_gdp = usa_employment_gdp.rename(columns={'PAYEMS':'美國非農業就業人數', 'UNRATE':'美國失業率', 'GDP':'美國GDP'})
usa_employment_gdp = usa_employment_gdp.resample('D').interpolate()

# 台灣 ------------------------------------------------------------------------

# 指數(台灣指數)
twii = yf.download('^TWII', start=start_date, end=end_date, interval='1d', progress=False)['Close'].rename('台灣指數')
# 公債殖利率
tw_bond_yeild_2y = pd.read_csv('data/tw_bond_2y.csv').set_index('日期')['收市'].rename('台灣2年期公債殖利率')
tw_bond_yeild_10y = pd.read_csv('data/tw_bond_10y.csv').set_index('日期')['收市'].rename('台灣10年期公債殖利率')
#tw_bond_yeild_30y = pd.read_csv('data/tw_bond_30y.csv').set_index('日期')['收市'].rename('台灣30年期公債殖利率')
tw_bond_yeild = pd.merge(tw_bond_yeild_2y, tw_bond_yeild_10y, left_index=True, right_index=True, how='outer')
#tw_bond_yeild = pd.merge(tw_bond_yeild, tw_bond_yeild_30y, left_index=True, right_index=True, how='outer')
tw_bond_yeild.index = pd.to_datetime(tw_bond_yeild.index)
# 基準利率(插植)
tw_interest_rate = pd.read_csv('data/tw_interest_rate.csv').set_index('日期')['台灣重貼現率']
tw_interest_rate.index = pd.to_datetime(tw_interest_rate.index)
tw_interest_rate = tw_interest_rate.resample('D').interpolate()
# CPI、PPI
tw_cpi_ppi = pd.read_csv('data/tw_cpi_ppi.csv').set_index('日期')
tw_cpi_ppi.index = pd.to_datetime(tw_cpi_ppi.index)
tw_cpi_ppi = tw_cpi_ppi.resample('D').interpolate()
# GDP、失業率、非農就業人數(插植)
tw_gdp = pd.read_csv('data/tw_gdp.csv').set_index('日期')
tw_gdp.index = pd.to_datetime(tw_gdp.index)
tw_gdp = tw_gdp.resample('D').interpolate()

tw_employment = pd.read_csv('data/tw_employment.csv').set_index('日期')
tw_employment.index = pd.to_datetime(tw_employment.index)
tw_employment = tw_employment.resample('D').interpolate()

# 政治穩定性(經濟自由度指數)
economic_freedom = pd.read_csv('data/economic_freedom.csv').set_index('日期')
economic_freedom.index = pd.to_datetime(economic_freedom.index)
economic_freedom = economic_freedom.resample('D').interpolate()

# 其他處理 ---------------------------------------------------------------------

# 合併資料
data = pd.merge(usdtwd, gspc, left_index=True, right_index=True, how='outer')
data = pd.merge(data, usa_bond_yield, left_index=True, right_index=True, how='outer')
data = pd.merge(data, usa_interest_rate, left_index=True, right_index=True, how='outer')
data = pd.merge(data, usa_cpi_ppi, left_index=True, right_index=True, how='outer')
data = pd.merge(data, usa_employment_gdp, left_index=True, right_index=True, how='outer')
data = pd.merge(data, twii, left_index=True, right_index=True, how='outer')
data = pd.merge(data, tw_bond_yeild, left_index=True, right_index=True, how='outer')
data = pd.merge(data, tw_interest_rate, left_index=True, right_index=True, how='outer')
data = pd.merge(data, tw_cpi_ppi, left_index=True, right_index=True, how='outer')
data = pd.merge(data, tw_gdp, left_index=True, right_index=True, how='outer')
data = pd.merge(data, tw_employment, left_index=True, right_index=True, how='outer')
data = pd.merge(data, economic_freedom, left_index=True, right_index=True, how='outer')
# 時間序列
data['時間序列']=range(len(data))
# 清理空值
data = data.dropna(axis=0)
data = data.iloc[1:]
# 匯出及輸出
data.to_csv('data/data.csv',encoding='utf_8_sig')
print(len(data))
print(data)

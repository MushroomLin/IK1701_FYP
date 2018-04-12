import quandl
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys

class conf:
    instrument = 'AAPL'
    table_start = '2007-12-01'
    table_end = '2018-02-01'
    start_date = '2008-01-01'
    end_date = '2018-01-01'
    fields = [ 'date', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume']  # features
    shift = 1

if len(sys.argv)>1: conf.instrument=sys.argv[1]
# Extract stock price data
quandl.ApiConfig.api_key = 'AH_8yF2qo1ShWwMn_uBr'
data = quandl.get_table('WIKI/PRICES',
                        ticker = [conf.instrument], date = { 'gte': conf.table_start, 'lte': conf.table_end })
print(data)
data=data[conf.fields]
# data['change'] is the percentage of stock price change
data['next_change'] = 100*(data['adj_close'].shift(-conf.shift) - data['adj_close'])/data['adj_close'].shift(-conf.shift)
data['change']= data['next_change'].shift(1)
data=data[conf.start_date<=data.date]
data=data[conf.end_date>data.date]
# Scale the data
sc=StandardScaler()
minmax=MinMaxScaler(feature_range=(-1,1))

scaled_data=sc.fit_transform(data[['adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume']])
scaled_data=pd.DataFrame(data=scaled_data,columns=['scaled_adj_close', 'scaled_adj_open', 'scaled_adj_high',
                                                   'scaled_adj_low', 'scaled_adj_volume'])
scaled_data['change']= np.array(data['change'])
scaled_data['next_change']=np.array(data['next_change'])
for i in conf.fields:
    scaled_data[i] = np.array(data[i])
scaled_data=scaled_data[[ 'date', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume',
                          'scaled_adj_close', 'scaled_adj_open', 'scaled_adj_high',
                          'scaled_adj_low', 'scaled_adj_volume','change','next_change']]
scaled_data.to_csv('./data/apple_standard.csv',index=False)



print(scaled_data)
scaled_data=minmax.fit_transform(data[['adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume']])
scaled_data=pd.DataFrame(data=scaled_data,columns=['scaled_adj_close', 'scaled_adj_open', 'scaled_adj_high',
                                                   'scaled_adj_low', 'scaled_adj_volume'])
for i in conf.fields:
    scaled_data[i] = np.array(data[i])
scaled_data['change']= np.array(data['change'])
scaled_data['next_change']=np.array(data['next_change'])
scaled_data=scaled_data[[ 'date', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume',
                          'scaled_adj_close', 'scaled_adj_open', 'scaled_adj_high',
                          'scaled_adj_low', 'scaled_adj_volume','change','next_change']]
print(scaled_data[['adj_close','change','next_change']])
scaled_data.to_csv('./data/apple_minmax.csv',index=False)
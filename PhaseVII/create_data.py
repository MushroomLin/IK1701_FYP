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
    split_date = '2016-01-01'
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
data['change'] = (data['adj_close'].shift(-conf.shift) - data['adj_open'].shift(-1))/data['adj_close'].shift(-conf.shift)
data=data[conf.start_date<=data.date]
data=data[conf.end_date>data.date]
# Scale the data
sc=StandardScaler()
scaled_data=sc.fit_transform(data[['adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume','change']])
scaled_data=pd.DataFrame(data=scaled_data,columns=['adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume','change'])
scaled_data['date']= np.array(data['date'])
scaled_data=scaled_data[[ 'date', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'adj_volume','change']]
print(scaled_data)
scaled_data.to_csv('./data/apple.csv',index=False)
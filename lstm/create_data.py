import quandl
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class conf:
    instrument = 'AAPL'
    table_start = '2007-12-01'
    table_end = '2018-02-01'

    start_date = '2008-01-01'
    split_date = '2016-01-01'
    end_date = '2018-01-01'
    fields = [ 'date', 'close', 'open', 'high', 'low', 'volume']  # features
    seq_len = 30 #length of input
    batch = 100
    shift = 1


# Extract stock price data
quandl.ApiConfig.api_key = 'AH_8yF2qo1ShWwMn_uBr'
data = quandl.get_table('WIKI/PRICES',
                        ticker = [conf.instrument], date = { 'gte': conf.table_start, 'lte': conf.table_end })
print(data)
data=data[conf.fields]
data['change'] = (data['close'].shift(-conf.shift) - data['open'].shift(-1))/data['close'].shift(-conf.shift)
data=data[conf.start_date<=data.date]
data=data[conf.end_date>data.date]
print(data)
sc=StandardScaler()
scaled_data=sc.fit_transform(data[['close', 'open', 'high', 'low', 'volume','change']])
scaled_data=pd.DataFrame(data=scaled_data,columns=['close', 'open', 'high', 'low', 'volume','change'])
scaled_data['date']= np.array(data['date'])
scaled_data=scaled_data[[ 'date', 'close', 'open', 'high', 'low', 'volume','change']]
print(scaled_data)
scaled_data.to_csv('./data/apple.csv',index=False)
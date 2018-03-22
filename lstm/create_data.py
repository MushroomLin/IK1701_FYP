import quandl
import numpy as np
import pandas as pd

class conf:
    instrument = 'AAPL'
    start_date = '2007-12-01'
    split_date = '2016-01-01'
    end_date = '2018-02-01'
    fields = ['close', 'open', 'high', 'low', 'amount', 'volume']  # features
    seq_len = 30 #length of input
    batch = 100


# Extract stock price data
quandl.ApiConfig.api_key = 'AH_8yF2qo1ShWwMn_uBr'
data = quandl.get_table('WIKI/PRICES',
                        ticker = [conf.instrument], date = { 'gte': conf.start_date, 'lte': conf.end_date })
data.from_records(data)
data.to_csv('./data/apple.csv')
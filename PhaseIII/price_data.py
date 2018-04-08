import quandl
import numpy as np
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = 'GZzqbzHAarxcAqdTwfZ7'
data = quandl.get_table('WIKI/PRICES',
                        ticker = ['AAPL', 'AMZN'], date = { 'gte': '2017-01-01', 'lte': '2017-10-31' })
data.from_records(data)
data.to_csv('./price.csv')
print(data)
date=np.array(data.loc[data['ticker'] == 'AAPL']['date'])
apple=np.array(data.loc[data['ticker'] == 'AAPL']['close'])
amazon=np.array(data.loc[data['ticker'] == 'AMZN']['close'])
plt.plot(date,apple,'r--')
plt.show()
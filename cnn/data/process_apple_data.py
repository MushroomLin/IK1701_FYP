import pandas as pd
import quandl
import numpy as np
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from datetime import timedelta, date, datetime

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


data=pd.read_csv('./apple_date.csv')
label=pd.read_csv('./prediction.csv')
label=label['Label']
t_data=pd.concat([data,label[:105408]],axis=1)
start_date = date(2017, 3, 1)
end_date = date(2017, 4, 1)
list_sentiment=np.zeros(31)
list_count=np.zeros(31)
day=0
list_date=[]
for single_date in daterange(start_date, end_date):
    date=single_date.strftime("%Y-%m-%d")
    print(date)
    list_date.append(single_date)
    for i,j in t_data.loc[t_data['Date']==date].iterrows():
        if(j['Label']==1):
            list_sentiment[day]+=1
        list_count[day]+=1
    day+=1
print(list_sentiment)
print(list_count)
list_average=list_sentiment/list_count

quandl.ApiConfig.api_key = 'GZzqbzHAarxcAqdTwfZ7'
data = quandl.get_table('WIKI/PRICES',
                        ticker = ['AAPL'], date = { 'gte': '2017-03-01', 'lte': '2017-03-31' })
data.from_records(data)
date=np.array(data.loc[data['ticker'] == 'AAPL']['date'])
apple=np.array(data.loc[data['ticker'] == 'AAPL']['close'])22

#
# plt.plot(date,apple,'r--')
# plt.title('APPL Stock Price')
# plt.show()
plt.plot(list_date,list_sentiment,'b--')
plt.plot(list_date,list_count,'g--')
plt.title('Daily positive news (blue) and Daily total news (green)')
plt.show()
plt.plot(list_date,list_average,'y--')
plt.title('Daily average sentiment')
plt.show()
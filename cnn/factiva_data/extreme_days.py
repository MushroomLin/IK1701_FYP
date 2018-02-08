# Find the extreme days news
import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, date, datetime

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


data=pd.read_csv('prediction.csv')
label=data['Sentiment']

start_date = date(2015, 1, 1)
start= start_date.strftime("%Y-%m-%d")
end_date = date(2016, 12, 31)
end= end_date.strftime("%Y-%m-%d")
total_day = (end_date-start_date).days
print("Total",total_day,"days.")

# Get number of positive news and total news
day=0
list_date=[]
list_sentiment=np.zeros(total_day)
list_count=np.zeros(total_day)

for single_date in daterange(start_date, end_date):
    date=single_date.strftime("%Y/%m/%d")
    print(date)
    list_date.append(single_date)
    for i,j in data.loc[pd.to_datetime(data['Date'])==pd.to_datetime(date)].iterrows():
        if(j['Sentiment']==1):
            list_sentiment[day]+=1
        list_count[day]+=1
    day+=1
list_average=list_sentiment/list_count


# Create Dataframe
data_frame={}
data_frame['Date']=list_date
data_frame['Total']=list_count
data_frame['Positive']=list_sentiment
data_frame['Average']=list_average

# Get 3 days average sentiment
index=0
average_three_days=[]
for i in list_average:
    if index==0 or index==1:
        average_three_days.append(i)
    else:
        average_three_days.append(np.mean([list_average[index-1],list_average[index-2],list_average[index]]))
    index+=1
average_three_days=np.array(average_three_days)

# Get sock price data
stock='AAPL'
quandl.ApiConfig.api_key = 'AH_8yF2qo1ShWwMn_uBr'

stock_data = quandl.get_table('WIKI/PRICES',
                        ticker = [stock], date = { 'gte': start, 'lte': end })
print(stock_data)
stock_data.from_records(stock_data)
trade_date=np.array(stock_data.loc[stock_data['ticker'] == stock]['date'])
apple=np.array(stock_data.loc[stock_data['ticker'] == stock]['close'])

trade_day=[]
for i in trade_date:
    trade_day.append(str(i)[:10])

price_list=[]
change_list=[]
volume_list=[]
for single_date in daterange(start_date, end_date):
    date=single_date.strftime('%Y-%m-%d')

    if date in trade_day:
        price=float(stock_data.loc[stock_data['date'] == date]['close'])
        change=float(stock_data.loc[stock_data['date'] == date]['close']-stock_data.loc[stock_data['date'] == date]['open'])
        volume=float(stock_data.loc[stock_data['date'] == date]['volume'])
    else:
        price='NaN'
        change='NaN'
        volume='NaN'
    price_list.append(price)
    change_list.append(change)
    volume_list.append(volume)

data_frame['Price']=price_list
data_frame['Change']=change_list
data_frame['Volume']=volume_list
df = pd.DataFrame(data=data_frame)
df = df[['Date','Total','Positive','Average','Volume','Price','Change']]
print(df)
df.to_csv('./sentiment.csv',index=False)
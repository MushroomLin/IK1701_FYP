import pandas as pd
import sklearn
data=pd.read_csv('./stocknews/Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
print(train.shape)
print(test.shape)
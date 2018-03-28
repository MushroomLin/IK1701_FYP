from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Sequential, load_model
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
import tensorflow as tf
def atan(x):
    return tf.atan(x)

def processData(data1,data2,lb):
    X,Y = [],[]
    for i in range(len(data1)-lb-1):
        X.append(data1[i:(i+lb)])
        Y.append(data2[i+lb])
    return np.array(X), np.array(Y)

def stock_algorithm(predictions, true):
    list_a=[]
    list_b=[]
    money=10000.0
    stock_price=100.0
    number=0
    cool_down = 0
    for i in range(len(predictions)):
        if predictions[i]>=0.03 and cool_down<=0:
            number+=money/stock_price
            money=0
            cool_down=10
        elif predictions[i]<0.03 and cool_down<=0:
            money+=number*stock_price
            number=0
            cool_down=10
        cool_down-=1
        list_a.append(money + stock_price * number)
        list_b.append(stock_price*100)
        stock_price+=true[i]
    money+=stock_price*number
    return money,stock_price,list_a,list_b

class conf:
    instrument = 'AAPL'
    start_date = '2008-01-01'
    split_date = '2016-01-01'
    end_date = '2018-01-01'
    fields = [ 'close', 'open', 'high', 'low', 'volume']  # features
    # fields=['open','volume']
    seq_len = 10 #length of input
    batch = 100
    shift = 1
    epochs = 100


data=pd.read_csv('./data/apple.csv')
# Return is the future 5 days profit


train = data[data.date<conf.split_date]
test = data[data.date>=conf.split_date]
test_date=test['date']
train_date=train['date']

print(train)
print(test)
test_x,test_y = processData(np.array(test[conf.fields]),np.array(test['change']),conf.seq_len)
train_x,train_y = processData(np.array(train[conf.fields]),np.array(train['change']),conf.seq_len)

# #Build the model
# model = Sequential()
# model.add(LSTM(128, dropout_W=0.2, dropout_U=0.1,input_shape=(conf.seq_len,len(conf.fields))))
# model.add(Dense(64,activation='linear'))
# model.add(Dense(16,activation='linear'))
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mse')
# #Fit model with history to check for overfitting
# history = model.fit(train_x,train_y,epochs=conf.epochs,shuffle=False)
#
# model.save('./lstm_model')

model=load_model('./lstm_model')
predictions = model.predict(test_x)
money,real,list_a,list_b=stock_algorithm(predictions,test_y)
print(money,real)
x=range(0,len(test_y))

date=list(test_date[:len(test_y)])
date=[dt.datetime.strptime(dat,'%Y-%m-%d').date() for dat in date]
plt.plot(date,predictions, 'o', label="prediction")
plt.plot(date,test_y,'x',label='ground truth')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.legend()
plt.show()


plt.plot(date,list_a,label='our algorithm')
plt.plot(date,list_b,label='overall')
plt.xlabel('Date')
plt.ylabel('Money')
plt.legend()
plt.show()
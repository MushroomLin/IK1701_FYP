from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Sequential

import matplotlib.pyplot as plt
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

class conf:
    instrument = 'AAPL'
    start_date = '2008-01-01'
    split_date = '2016-01-01'
    end_date = '2018-01-01'
    fields = [ 'close', 'open', 'high', 'low', 'volume']  # features
    seq_len = 30 #length of input
    batch = 100
    shift = 5


data=pd.read_csv('./data/apple.csv')
# Return is the future 5 days profit
data['change'] = data['close'].shift(-conf.shift) - data['open'].shift(-1)
data=data[conf.start_date<=data.date]
data=data[conf.end_date>data.date]


train = data[data.date<conf.split_date]
test = data[data.date>=conf.split_date]

sc=StandardScaler()

scale_data_train = np.array(train[conf.fields])
change_train= np.array(train['change'])
scale_data_test = np.array(test[conf.fields])
change_test= np.array(test['change'])
sc.fit(np.vstack((scale_data_train,scale_data_test)))
scale_data_train=sc.transform(scale_data_train)
scale_data_test=sc.transform(scale_data_test)

sc2=StandardScaler()
sc2.fit(np.concatenate((change_train,change_test)))
change_train=sc2.transform(change_train)
change_test=sc2.transform(change_test)

test_x,test_y = processData(scale_data_test,change_test,conf.seq_len)
train_x,train_y = processData(scale_data_train,change_train,conf.seq_len)

print(train_x)

#Build the model
model = Sequential()
model.add(LSTM(128, activation=atan, dropout_W=0.2, dropout_U=0.1,input_shape=(30,5)))
model.add(Dense(64,activation='linear'))
model.add(Dense(16,activation='linear'))
model.add(Dense(1,activation=atan))
model.compile(optimizer='adam',loss='mse')
#Fit model with history to check for overfitting
history = model.fit(train_x,train_y,epochs=100,validation_data=(test_x,test_y),shuffle=False)

predictions = model.predict(test_x)
print(predictions)
x=range(0,len(test_y))
plt.plot(x,predictions, 'o', label="prediction")
plt.plot(x,test_y,'x',label='ground truth')
plt.show()

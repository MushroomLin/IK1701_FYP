from keras.layers import Input, Dense, LSTM, merge,LeakyReLU
from keras.models import Sequential, load_model
from keras.utils import plot_model
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K

def processData(data1,data2,lb):
    X,Y = [],[]
    for i in range(len(data1)-lb-1):
        X.append(data1[i:(i+lb)])
        Y.append(data2[i+lb])
    return np.array(X), np.array(Y)

def stock_algorithm(predictions, true):
    list_a=[]
    list_b=[]
    true=list(true)
    money=10000.0
    stock_price=true[0]
    number=0
    for i in range(len(predictions)):
        if predictions[i]>=0:
            number+=money/stock_price
            money=0
        elif predictions[i]<0:
            money+=number*stock_price
            number=0
        print(stock_price)
        list_a.append(money + stock_price * number)
        list_b.append(stock_price*(10000/true[0]))
        stock_price=true[i+1]
    money+=stock_price*number
    return money,stock_price,list_a,list_b

def accuracy(predictions, test_y):
    average=sum(test_y) / float(len(test_y))
    correct = 0
    total = 0
    for i in range(len(predictions)):
        if (predictions[i] > average and test_y[i] > average) or (predictions[i] <= average and test_y[i] <= average):
            correct += 1
        total += 1
    print(total, correct)
    print('Accuracy %f' % float(correct / total))

class conf:
    instrument = 'AAPL'
    start_date = '2008-01-01'
    split_date = '2016-01-01'
    end_date = '2018-01-01'
    fields = [ 'change']  # features
    seq_len = 10 #length of input
    batch = 128 # batch size
    epochs = 200 # num of epochs to train
    scale ='minmax' # way of scale training data, either minmax or standard


data=pd.read_csv('./data/apple_'+conf.scale+'.csv')
# Return is the future 5 days profit


train = data[data.date<conf.split_date]
test = data[data.date>=conf.split_date]
test_date=test['date']
train_date=train['date']


test_x,test_y = processData(np.array(test[conf.fields]),np.array(test['next_change']),conf.seq_len)
train_x,train_y = processData(np.array(train[conf.fields]),np.array(train['next_change']),conf.seq_len)

# Build the model
model = Sequential()
model.add(LSTM(128, dropout=0,input_shape=(conf.seq_len,len(conf.fields))))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Fit model with history to check for overfitting
history = model.fit(train_x,train_y,epochs=conf.epochs,batch_size=conf.batch,shuffle=False,validation_split=0.1)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./graph/loss.png')
plt.show()

# Save the model
model.save('./lstm_model')
# Load the model
model=load_model('./lstm_model')

# Predict on training set and draw return graph
predictions = model.predict(test_x)
predictions_X = model.predict(train_x)
# Check for training
money,real,list_a,list_b=stock_algorithm(predictions_X,train['adj_close'])
print(money,real)
date_train=list(train_date[:len(train_y)])
date_train=[dt.datetime.strptime(dat,'%Y-%m-%d').date() for dat in date_train]
plt.scatter(date_train,predictions_X, s=3, label="prediction")
plt.scatter(date_train,train_y,s=3,label='ground truth')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('Train price change and true value')
plt.legend()
plt.savefig('./graph/train_prediction_'+str(conf.epochs)+'.png')
plt.show()

plt.plot(date_train,list_a,label='our algorithm')
plt.plot(date_train,list_b,label='overall')
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Train Return')
plt.legend()
plt.savefig('./graph/train_return_'+str(conf.epochs)+'.png')
plt.show()

# Predict on testing set and draw return graph
money,real,list_a,list_b=stock_algorithm(predictions,test['adj_close'])
print(money,real)

date_test=list(test_date[:len(test_y)])
date_test=[dt.datetime.strptime(dat,'%Y-%m-%d').date() for dat in date_test]
plt.scatter(date_test,predictions, s=5, label="prediction")
plt.scatter(date_test,test_y,s=5,label='ground truth')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('Test price change and true value')
plt.legend()
plt.savefig('./graph/test_prediction_'+str(conf.epochs)+'.png')
plt.show()

plt.plot(date_test,list_a,label='our algorithm')
plt.plot(date_test,list_b,label='overall')
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Test Return')
plt.legend()
plt.savefig('./graph/test_return_'+str(conf.epochs)+'.png')
plt.show()
# Accuracy
accuracy(predictions,test_y)
accuracy(predictions_X,train_y)
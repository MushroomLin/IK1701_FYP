import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
import csv
import config
import math

prices_dataset =  pd.read_csv(config.DATASET, header=0)
company = prices_dataset[prices_dataset['symbol']==config.COMPANY]
# For AAPL
if config.COMPANY == 'AAPL':
    apple = company.values
    for line in apple:
        if line[0] <= '2014-06-06':
            line[3] = line[3] / 7
    company = pd.DataFrame(np.array(apple), columns=['date','symbol','open','close','low','high','volume'])
stock_prices = company.close.values.astype('float32')
date = company.date.values
stock_prices = stock_prices.reshape(-1,1)
# print apple_stock_prices.shape
scaler = MinMaxScaler(feature_range=(0,1))
stock_prices = scaler.fit_transform(stock_prices)
# print apple_stock_prices
train_size = int(len(stock_prices) * config.TRAIN_RATIO)
test_size = len(stock_prices) - train_size
train, test = stock_prices[0:train_size,:],stock_prices[train_size:len(stock_prices), :]
train_date, test_date = date[0:train_size],date[train_size:len(stock_prices)]
# print(len(train),len(test))
# print(train[0:2,0])

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, config.TIME_SPAN)
testX, testY = create_dataset(test, config.TIME_SPAN)

trainX = np.reshape(trainX, (trainX.shape[0],  trainX.shape[1],1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1],1))

#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    input_length=config.TIME_SPAN,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('compilation time : ', time.time() - start)


model.fit(
    trainX,
    trainY,
    batch_size=128,
    nb_epoch=10,
    validation_split=0.05)
    
#predict length consecutive values from a real one
def predict_sequences_multiple(model, firstValue,length):
    prediction_seqs = []
    curr_frame = firstValue 
    for i in range(length): 
        predicted = []
        # print(model.predict(curr_frame[newaxis,:,:]))
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[0:]
        curr_frame = np.insert(curr_frame[0:], i+1, predicted[-1], axis=0)
        prediction_seqs.append(predicted[-1])
    return prediction_seqs

def loss(true, predict):
    return np.square(abs(true-predict) / true)

predict_length=1
testY_origin = scaler.inverse_transform(np.array(testY).reshape(-1,1))
trainY_origin = scaler.inverse_transform(np.array(trainY).reshape(-1,1))
train_loss = 0
test_loss = 0
# with open(config.COMPANY + '_plot_' + str(config.TIME_SPAN) +'.csv', 'wb') as csvfile: 
#     writer = csv.writer(csvfile)
#     writer.writerow(['Date', 'Real', 'Predict'])
for i in range(len(trainX)):
    predictions = predict_sequences_multiple(model, trainX[i], predict_length)
    true_value = trainY_origin[i,0]
    predict_value = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[0,0]
    # writer.writerow([train_date[i+config.TIME_SPAN], true_value, predict_value])
    train_loss += loss(true_value, predict_value)
train_loss = train_loss / len(trainX)
train_loss = train_loss / math.sqrt(config.TIME_SPAN)
for i in range(len(testX)):
    predictions = predict_sequences_multiple(model, testX[i], predict_length)
    true_value = testY_origin[i,0]
    predict_value = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[0,0]
    # writer.writerow([test_date[i+config.TIME_SPAN], true_value, predict_value])
    test_loss += loss(true_value, predict_value)
test_loss = test_loss / math.sqrt(config.TIME_SPAN)

with open('loss.txt', 'a') as file:
    file.write(str(config.TIME_SPAN) + '    '+str(train_loss) + '    ' + str(test_loss) + '\r\n')

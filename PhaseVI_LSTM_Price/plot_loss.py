import matplotlib.pyplot as plt
from scipy.interpolate import spline 
import numpy as np
import pandas as pd
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

def loss(true, predict):
    return np.square(abs(true-predict) / true)

avg_loss = []
for i in range(1,31):
    loss_value = 0
    count = 0
    # print stock_prices.shape[0]
    for j in range(i, stock_prices.shape[0]):
        true_value = stock_prices[j]
        predict_value = 0
        for k in range(j-i,j):
            predict_value += stock_prices[k]
        predict_value = predict_value / i
        loss_value += loss(true_value, predict_value)
        count += 1
    loss_value = loss_value / count
    loss_value = loss_value / math.sqrt(i)
    avg_loss.append(loss_value)

index = []
train_loss = []
test_loss = []
with open('loss.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        index.append(int(line.split()[0]))
        train_loss.append(float(line.split()[1]))
        test_loss.append(float(line.split()[2]))

index = np.array(index)
train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
avg_loss = np.array(avg_loss)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
xnew = np.linspace(index.min(),index.max(),10000)
train_new = spline(index, train_loss, xnew)
test_new = spline(index, test_loss, xnew)
avg_new = spline(index, avg_loss, xnew)
plt.plot(xnew, train_new, 'b',label='Train')
plt.plot(xnew, test_new, 'r', label='Test')
plt.plot(xnew, avg_new, 'g',label='Baseline')
plt.xlabel('Time Span (days)')
plt.ylabel('Loss')
ax.set_xticks([0,5,10,15,20,25,30])
# ax.set_yticks([0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
# ax.set_yticks([0, 1e-3, 2e-3])
# plt.title('LSTM loss variation with time span')
plt.legend()
plt.show()
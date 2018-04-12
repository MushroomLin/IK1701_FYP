# from keras.layers import Input, Dense, LSTM, merge
# from keras.models import Sequential, load_model
# from keras.utils import plot_model
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def stock_algorithm(predictions, true, threhold=0):
    print('*********************************')
    list_a=[]
    list_b=[]
    true=list(true)
    money=10000.0
    length=len(true)-len(predictions)
    stock_price=true[length-1]
    number=0
    for i in range(len(predictions)):
        if predictions[i]>=threhold:
            number+=money/stock_price
            money=0
        elif predictions[i]<threhold:
            money+=number*stock_price
            number=0
        list_a.append(money + stock_price * number)
        list_b.append(stock_price*(10000/true[length-1]))
        stock_price=true[i+length]
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
    adjust_range = 0.05
    adjust_value = 0.5
    news_number = 50

sentiment=pd.read_csv('./data/sentiment.csv')
lstm=pd.read_csv('./data/lstm_prediction.csv')
sentiment=sentiment.merge(lstm)
print(sentiment)
combined=[]
day_plus=0
day_minus=0
mean_sentiment=sentiment['Average'].mean()

for index, data in sentiment.iterrows():
    if data['Total']>conf.news_number and data['Average']>mean_sentiment+conf.adjust_range:
        combined.append(data['Prediction']+conf.adjust_value)
        day_plus+=1
    elif data['Total']>conf.news_number and data['Average']<mean_sentiment-conf.adjust_range:
        combined.append(data['Prediction']-conf.adjust_value)
        day_minus+=1
    else:
        combined.append(data['Prediction'])

print('Total days: %d, Changed Plus days: %d, Changed Minus days: %d'%(len((combined)),day_plus, day_minus))

test_date=sentiment['Date']
# Predict on testing set and draw return graph
money,real,list_a,list_b=stock_algorithm(combined,sentiment['Price'])
print(money,real)

date_test=list(test_date)
date_test=[dt.datetime.strptime(dat,'%Y-%m-%d').date() for dat in date_test]
# Draw graph for combined model
fig1, ax1 = plt.subplots()
plt.scatter(date_test,combined, s=5, label="prediction")
plt.scatter(date_test,sentiment['Change'],s=5,label='ground truth')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('Test price change and true value')
plt.legend()
fig1.autofmt_xdate()
plt.savefig('./graph/combined_test_prediction.png')
plt.show()


fig2, ax2 = plt.subplots()
plt.plot(date_test,list_a,label='our algorithm')
plt.plot(date_test,list_b,label='overall')
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Test Return')
plt.legend()
fig2.autofmt_xdate()
plt.savefig('./graph/combined_test_return.png')
plt.show()


# Predict on testing set and draw return graph
money,real,list_a,list_b=stock_algorithm(sentiment['Average'],sentiment['Price'],mean_sentiment)
print(money,real)

# Draw graph for sentiment only
fig3, ax3 = plt.subplots()
plt.scatter(date_test,(sentiment['Average']-mean_sentiment)*10, s=5, label="prediction")
plt.scatter(date_test,sentiment['Change'],s=5,label='ground truth')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('Test price change and true value')
plt.legend()
fig3.autofmt_xdate()
plt.savefig('./graph/sentiment_test_prediction.png')
plt.show()


fig4, ax4 = plt.subplots()
plt.plot(date_test,list_a,label='our algorithm')
plt.plot(date_test,list_b,label='overall')
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Test Return')
plt.legend()
fig4.autofmt_xdate()
plt.savefig('./graph/sentiment_test_return.png')
plt.show()


# Draw graph for random only
rand=np.random.rand(len(sentiment))*2-1
money,real,list_a,list_b=stock_algorithm(rand,sentiment['Price'])
print(money,real)

fig5, ax5 = plt.subplots()
plt.scatter(date_test,rand, s=5, label="prediction")
plt.scatter(date_test,sentiment['Change'],s=5,label='ground truth')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('Test price change and true value')
plt.legend()
fig5.autofmt_xdate()
plt.savefig('./graph/random_test_prediction.png')
plt.show()


fig6, ax6 = plt.subplots()
plt.plot(date_test,list_a,label='our algorithm')
plt.plot(date_test,list_b,label='overall')
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Test Return')
plt.legend()
fig6.autofmt_xdate()
plt.savefig('./graph/random_test_return.png')
plt.show()


# Draw graph for lstm only
money,real,list_a,list_b=stock_algorithm(sentiment['Prediction'],sentiment['Price'])
print(money,real)
fig7, ax7 = plt.subplots()
plt.scatter(date_test,sentiment['Prediction'], s=5, label="prediction")
plt.scatter(date_test,sentiment['Change'],s=5,label='ground truth')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('Test price change and true value')
plt.legend()
fig7.autofmt_xdate()
plt.savefig('./graph/lstm_test_prediction.png')
plt.show()


fig8, ax8 = plt.subplots()
plt.plot(date_test,list_a,label='our algorithm')
plt.plot(date_test,list_b,label='overall')
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Test Return')
plt.legend()
fig8.autofmt_xdate()
plt.savefig('./graph/lstm_test_return.png')
plt.show()

sent=sentiment['Average']-mean_sentiment
# Accuracy
accuracy(combined,sentiment['Change'])
accuracy(sent,sentiment['Change'])
accuracy(sentiment['Prediction'],sentiment['Change'])
import pandas as pd
import sklearn
import nltk
import data_helper as dh
from nltk.corpus import sentiwordnet as swn
# Read data
data=pd.read_csv('./stocknews/Combined_News_DJIA.csv')
# Shape: [Date, Label, Top1-25 News]
# Label: "1" when DJIA Adj Close value rose or stayed as the same;
#        "0" when DJIA Adj Close value decreased.
print(data.columns.values)
# Divide data into train and test part
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
num_train_samples=train.shape[0]
num_test_samples=test.shape[0]

# Do data pre-process
# Only use Top1 of every day, Todo: add all news into string
train_trans=[]
for i in train.ix[:,2]:
    i=dh.clean_str(i)
    train_trans.append(nltk.word_tokenize(i))
print(train_trans)
test_trans=[]
for i in test.ix[:,2]:
    i=dh.clean_str(i)
    train_trans.append(nltk.word_tokenize(i))
print(test_trans)

# Todo: Convert train/test_trans into sentiment matrix train_X, test_X


train_label=train['label']
test_label=test['label']
clf=sklearn.svm.SVC()
clf.fit(train_X,train_label)
print('accuracy =',clf.score(test_X,test_label))
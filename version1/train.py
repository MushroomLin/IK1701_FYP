import pandas as pd
import nltk
import data_helper as dh
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sentiwordnet import Sentiwordnet

L = 20 #length of a news title

# Read data
data=pd.read_csv('../stocknews/Combined_News_DJIA.csv')
# Shape: [Date, Label, Top1-25 News]
# Label: "1" when DJIA Adj Close value rose or stayed as the same;
#        "0" when DJIA Adj Close value decreased.

#print("Colums: %s" % data.columns.values)

# Divide data into train and test part
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
num_train_samples=train.shape[0]
num_test_samples=test.shape[0]

print("Number of training samples: %d" % num_train_samples)
print("Number of testing samples: %d" % num_test_samples)

# Do data pre-process
# Only use Top1 of every day, Todo: add all news into string
# trans = sentiment feature
# trans2 = countvectorizer feature
train_trans=[]
train_trans2=[]
cv=CountVectorizer()
for row in range(0,len(train.index)):
    new_str=' '.join(str(x) for x in train.iloc[row,2:27])
    train_trans2.append(new_str)
    train_trans.append(nltk.word_tokenize(new_str))
train_trans2=cv.fit_transform(train_trans2)


# Transform1
test_trans=[]
test_trans2=[]
for row in range(0,len(test.index)):
    new_str=' '.join(str(x) for x in test.iloc[row,2:27])
    test_trans2.append(new_str)
    test_trans.append(nltk.word_tokenize(new_str))
test_trans2=cv.transform(test_trans2)

train_Y=train['Label']
test_Y=test['Label']


'''
sentiwordnet = Sentiwordnet()
#Convert train/test_trans into sentiment matrix train_X, test_X
train_input=[]
for sentence in train_trans:
	score_vec=[]
	for i in range(L):
		if i >= len(sentence):
			score_vec.append(float(0))
		else:
			score = sentiwordnet.get_sentiment(sentence[i])
			score_vec.append(score)
	train_input.append(score_vec)
train_X = np.array(train_input)
#print(train_X)
test_input=[]
for sentence in test_trans:
	score_vec=[]
	for i in range(L):
		if i >= len(sentence):
			score_vec.append(float(0))
		else:
			score = sentiwordnet.get_sentiment(sentence[i])
			score_vec.append(score)
	test_input.append(score_vec)
test_X = np.array(test_input)
'''
# Use Sentiwordnet
sentiwordnet = Sentiwordnet()
train_input=[]
for sentence in train_trans:
    score = 0
    for word in sentence:
        score += sentiwordnet.get_sentiment(word)
    train_input.append(score)
train_X = np.reshape(np.array(train_input), [-1,1])
test_input=[]
for sentence in test_trans:
    score = 0
    for word in sentence:
        score += sentiwordnet.get_sentiment(word)
    test_input.append(score)
test_X = np.reshape(np.array(test_input), [-1,1])

print(train_X.shape)
print(test_X.shape)
num_p=test_Y.tolist().count(1)
num_n=test_Y.tolist().count(0)
print(num_p/(num_n+num_p))
#Training
# Logistic Regression
lr = LogisticRegression()
lr.fit(train_trans2,train_Y)
print(lr.predict(test_trans2))
print('VC feature LR accuracy = ',lr.score(test_trans2,test_Y))
clf2 = svm.SVC()
clf2.fit(train_X,train_Y)
print(clf2.predict(test_X))
print('Sentiment feature SVM accuracy =',clf2.score(test_X, test_Y))
# Logistic Regression
lr2 = LogisticRegression()
lr2.fit(train_X,train_Y)
print(lr2.predict(test_X))
print(lr2.coef_)
print('Sentiment feature LR accuracy = ',lr2.score(test_X,test_Y))
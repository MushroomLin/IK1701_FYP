import pandas as pd
import numpy as np
import nltk
import data_helper as dh
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sentiwordnet import Sentiwordnet
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
cv=CountVectorizer(ngram_range=(2,2))
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
train_X=[]
for sentence in train_trans:
    score = 0
    for word in sentence:
        score += sentiwordnet.get_sentiment(word)
    train_X.append(score)


test_X=[]
for sentence in test_trans:
    score = 0
    for word in sentence:
        score += sentiwordnet.get_sentiment(word)
    test_X.append(score)


# Change train_X and test_X to be previous 3 days sentiment
# new_train_X=[]
# for i in range(len(train_X)):
#     if i>=2:
#         new_train_X.append([train_X[i],train_X[i-1],train_X[i-2]])
#     elif i==1:
#         new_train_X.append([train_X[i], train_X[i - 1], 0])
#     elif i==0:
#         new_train_X.append([train_X[i], 0, 0])
# train_X=new_train_X
# # train_X=np.mean(train_X,axis=1)
#
# new_test_X=[]
# for i in range(len(test_X)):
#     if i>=2:
#         new_test_X.append([test_X[i],test_X[i-1],test_X[i-2]])
#     elif i==1:
#         new_test_X.append([test_X[i], test_X[i - 1], 0])
#     elif i==0:
#         new_test_X.append([test_X[i], 0, 0])
# test_X=new_test_X
# test_X=np.mean(test_X,axis=1)


# Reshape data
train_X = np.array(train_X)
test_X = np.array(test_X)
train_X=train_X.reshape(-1,1)
test_X = test_X.reshape(-1,1)

# Look at label ratio in the test dataset
num_p=test_Y.tolist().count(1)
num_n=test_Y.tolist().count(0)
print('in test dataset positive ratio',num_p/(num_n+num_p))
#Training
# SVM
clf = svm.SVC()
clf.fit(train_trans2,train_Y)
print('Count vectorize feature SVM accuracy = ',clf.score(test_trans2,test_Y))
# Logistic Regression
lr = LogisticRegression()
lr.fit(train_trans2,train_Y)
print('Count vectorizefeature LR accuracy = ',lr.score(test_trans2,test_Y))
# Random Forest
rf = RandomForestClassifier()
rf.fit(train_trans2,train_Y)
print('Count vectorize feature RF accuracy = ',rf.score(test_trans2,test_Y))
# SGD
sgd= SGDClassifier()
sgd.fit(train_trans2,train_Y)
print('Count vectorize feature SGD accuracy = ',sgd.score(test_trans2,test_Y))
# Model with sentiment analysis
# SVM
clf2 = svm.SVC()
clf2.fit(train_X,train_Y)
predict_Y=clf2.predict(test_X)
print('Sentiment feature SVM accuracy =',clf2.score(test_X, test_Y))
# Logistic Regression
lr2 = LogisticRegression()
lr2.fit(train_X,train_Y)
print('Sentiment feature LR accuracy = ',lr2.score(test_X,test_Y))
# Random Forest
rf2 = RandomForestClassifier()
rf2.fit(train_X,train_Y)
print('Sentiment feature RF accuracy = ',rf2.score(test_X,test_Y))
# SGD
sgd2= SGDClassifier()
sgd2.fit(train_X,train_Y)
print('Sentiment feature SGD accuracy = ',sgd2.score(test_X,test_Y))

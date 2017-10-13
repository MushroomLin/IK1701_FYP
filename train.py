import pandas as pd
import nltk
import data_helper as dh
import numpy as np
from sklearn import svm
from sentiwordnet import Sentiwordnet

L = 20 #length of a news title

# Read data
data=pd.read_csv('./stocknews/Combined_News_DJIA.csv')
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
train_trans=[]
for i in train.ix[:,2]:
    i=dh.clean_str(i)
    train_trans.append(nltk.word_tokenize(i))
#print(train_trans)
test_trans=[]
for i in test.ix[:,2]:
    i=dh.clean_str(i)
    test_trans.append(nltk.word_tokenize(i))
#print(test_trans)
train_label=train['Label']
test_label=test['Label']


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


#Convert train/test_label into np array train_Y, test_Y
train_label_array=[]
for i in train_label:
	train_label_array.append(i)
train_Y=np.array(train_label_array)
test_label_array=[]
for i in test_label:
	test_label_array.append(i)
test_Y=np.array(test_label_array)

#Training
clf = svm.SVC()
clf.fit(train_X,train_Y)
print(clf)
print('accuracy =',clf.score(test_X, test_Y))

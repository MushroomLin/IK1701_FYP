import time            
import re            
import os    
import sys  
import csv
#import codecs  
#import shutil  
import pandas as pd
import numpy as np
from sklearn import feature_extraction   
from sklearn.cluster import KMeans  
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer

DATASET_NAME = "Apple_News_Data"
OUTPUT_FILE_NAME = DATASET_NAME + "_Clustered"
KEY = "Title"

if __name__ == "__main__":
    df = pd.read_csv(DATASET_NAME+'.csv',header=0)
    
    corpus = []
    for i in range(len(df)):
        s = df[KEY].iloc[i].strip()
        s = unicode(s, errors='ignore')
        corpus.append(s)

    vectorizer = CountVectorizer(stop_words='english')  
    transformer = TfidfTransformer() 
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()  
    weight = tfidf
    
    clf = KMeans(n_clusters=3)  
    s = clf.fit(weight)

    # for i in range(weight.shape[0]):  
    #     for j in range(len(word)):  
    #         print str(weight[i][j]),
    
    # num = [0,0,0]
    # for i in clf.labels_:
    #     num[i] += 1
    # print num

    reader = csv.reader(open(DATASET_NAME+'.csv', 'rb'))
    writer = csv.writer(open(OUTPUT_FILE_NAME+'.csv', 'wb'))
    headers = reader.next()
    headers.append("Label")
    writer.writerow(headers)
    ix = 0
    for row in reader:
        row.append(clf.labels_[ix])
        writer.writerow(row)
        ix += 1
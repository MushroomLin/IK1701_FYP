import pandas as pd
import re
def clean_str(string):

    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string=string.lower()

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

data=pd.read_csv('./Twitter.csv',encoding='latin-1',names=['target','ids','date','flag','user','text'])
pos_data=data.loc[data['target']==4]['text']
neg_data=data.loc[data['target']==0]['text']
print(pos_data)
f_pos=open('./positive','w+')
f_neg=open('./negative','w+')
k=0
for i in pos_data:
    f_pos.write(clean_str(i)+'\n')
    k+=1
w=0
for i in neg_data:
    f_neg.write(clean_str(i)+'\n')
    w+=1
    if (w==k): break
f_pos.close()
f_neg.close()
print(w)
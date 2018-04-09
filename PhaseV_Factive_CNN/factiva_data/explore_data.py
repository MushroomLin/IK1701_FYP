import pandas as pd

data=pd.read_csv('./sentiment.csv')
data=data.dropna()
data['3_average']=(data['Average'].shift(1)+data['Average'].shift(2)+data['Average'])/3
print(data['3_average'])

print(data['Average'])
total=0
pos=0

for i in data['Total']:
    total+=i
for i in data['Positive']:
    pos+=i
average=pos/total
print(average)
true=0
false=0
for index,row in data.iterrows():
    if (row['Change']>0 and row['3_average']>average) or (row['Change']<=0 and row['3_average']<=average):
        true+=1
    elif (row['Change']>0 and row['3_average']<average) or (row['Change']<=0 and row['3_average']>=average):
        false+=1
print(true)
print(false)
print(true/(true+false))

true=0
false=0
for index,row in data.iterrows():
    if (row['Change']>0 and row['Average']>average) or (row['Change']<=0 and row['Average']<=average):
        true+=1
    elif (row['Change']>0 and row['Average']<average) or (row['Change']<=0 and row['Average']>=average):
        false+=1
print(true)
print(false)
print(true/(true+false))
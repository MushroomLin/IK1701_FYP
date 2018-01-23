import pandas as pd

f=pd.read_csv('./facebook_data.csv')
print(f['Title'])

new_file=open('./facebook.csv','w+')
for i in f['Title']:
    print(i)
    new_file.write(str(i)+'\n')
new_file.close()
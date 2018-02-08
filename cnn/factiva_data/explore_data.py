import pandas as pd

data=pd.read_csv('./sentiment.csv')
print(data.sort_values(by='Total'))
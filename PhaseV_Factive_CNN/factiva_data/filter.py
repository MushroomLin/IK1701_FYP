import pandas as pd
import numpy as np
import datetime
import os
import math
import sys

KEY = sys.argv[1]

def filter(key, handle_NAN=False):
    df = pd.read_csv('statistics.csv',header=0)

    if handle_NAN:
        handle = []
        for i in range(len(df)):
            value = df[key].iloc[i]
            if math.isnan(value):
                handle.append(0)
            elif value < 0:
                handle.append(-value)
            else:
                handle.append(value)
        s = pd.Series(np.array(handle))
        df = df.assign(abs=s.values)
        sorted_df = df.sort_values(by='abs', ascending=False)
    else:
        sorted_df = df.sort_values(by=key, ascending=False)

    # Target dates
    dates = sorted_df.head(int(len(df) * 0.1))['Date'].sort_values()

    news_df = pd.read_csv('Apple_News_Data.csv',header=0)
    news_df = news_df.set_index(['Date'])
    df = df.set_index(['Date'])

    if os.path.exists(key+'.csv'):
        return
    with open(key+'.csv','a') as f:
        header = False
        for i in range(len(dates)):
            dt = datetime.datetime.strptime(dates.iloc[i], '%Y-%m-%d')
            dt = '{0}/{1}/{2}'.format(dt.year, dt.month, dt.day)
            if df.loc[dates.iloc[i]].Total < 10:
                continue
            if header == False:
                news_df.loc[dt].to_csv(f, index=True, header=True)
                header = True
            else:
                news_df.loc[dt].to_csv(f, index=True, header=False)

if KEY != 'Change':
    filter(KEY)
else:
    filter(KEY, handle_NAN=True)
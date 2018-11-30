# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from urllib.request import urlopen

df  = pd.read_csv('cik_list.csv')
valid_df = df.iloc[:152,:].values
ur = valid_df[0][5]

for i in range(152):
    valid_df[i][5]="https://www.sec.gov/Archives/"+valid_df[i][5]
    print(valid_df[0][5])

full_data=[]
data = pd.read_csv(ur, skiprows=4, header=None, sep='\s+')
full_data.append(data)

for i in range(152):
    url=valid_df[i][5]
    #print(url)
    data = pd.read_csv(url)
    full_data.append(data)


valid_df[0][5]

url ='http://www2.conectiv.com/cpd/tps/archives/nj/2017/12/20171205NJA1.txt'
data = pd.read_csv(url, skiprows=4, header=None, sep='\s+')
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:54:53 2018

@author: dhruv
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv',low_memory=False)

dataset.head().transpose
dataset.shape
corpus = []
corpus.append(dataset.dtypes)
corpus.append(dataset.isnull().sum())

dataset.dtypes[42]

np.array(corpus)
dataset.dtypes
dataset.isnull().sum()
index =dataset.isnull().sum()
np.asarray(index)
j=-1

total_null=[]
int_only =[]

for i in index:
    j+=1
    if i>0 | (dataset.dtypes[j]!=int):
        total_null.append(j)
        
k=-1     
for i in index:
    k+=1
    if i>0:
        int_only.append(k)

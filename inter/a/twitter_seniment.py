# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:34:57 2018

@author: dhruv
"""

import pandas as pd
import numpy as np
import re

dataset = pd.read_csv('train.tsv', delimiter= '\t', quoting =3)
#df1 = pd.read_csv('test.csv')

dataset.head(20)
dataset.shape

dataset.isnull().sum()

dataset.describe()

# clearing the datset
X = dataset
X =np.asarray(X)
corpus =[]

for i in range(len(X)):
    review = re.sub(r'\W',' ',str(X[i][2]))
    review = re.sub(r'[0-9]',' ',review)
    #review = re.sub(r'^\s+',' ',review)
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    #review = re.sub(r'[a-z]\s+',' ',review)
    #review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'[ð½¼âïã¾º]',' ',review)
    review = re.sub(r'/(?<!\S).(?!\S)\s*/','',review)
    review = re.sub(r'^\s+', '', review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)


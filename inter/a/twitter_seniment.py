# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:34:57 2018

@author: dhruv
"""

import pandas as pd
import numpy as np
import re

dataset = pd.read_csv('train.tsv', delimiter= '\t', quoting =3)
df1 = pd.read_csv('t.tsv',delimiter= '\t', quoting =3)

dataset.head(20)
dataset.shape

dataset.isnull().sum()

dataset.describe()

# clearing the datset
X = dataset
X =np.asarray(X)
X1 = df1
X1=np.asarray(X1)
corpus_train =[]
corpus_test = []

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
    corpus_train.append(review)
i=0
for i in range(len(X1)):
    review = re.sub(r'\W',' ',str(X1[i][1]))
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
    corpus_test.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x_train = cv.fit_transform(corpus_train).toarray()
y_train = dataset.iloc[:,1].values

cv1 = CountVectorizer(max_features = 1500)
x_test = cv1.fit_transform(corpus_test).toarray()
#y_test = df1.iloc[:,1].values

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

z = df1.iloc[:,0].values
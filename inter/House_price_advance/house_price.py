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

dataset.dtypes[4]

np.array(corpus)
dataset.dtypes
dataset.isnull().sum()
index = list(dataset.isnull().sum())
np.asarray(index)
int_only =[]
k=-1     
for i in index:
    k+=1
    if i>0:
        int_only.append(k)
X_train= dataset.iloc[:,0:80]

X_train = X_train.drop(['Alley', 'MasVnrType','BsmtQual','BsmtCond','FireplaceQu','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageFinish','GarageQual','GarageCond','Fence','MiscFeature'],axis=1)


index[1]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X_train[:, [3,24,49]])
X_train[:, [3,24,49]] = imputer.transform(X_train[:,[3,24,49]])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:,[]] = labelencoder_X.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_train = onehotencoder.fit_transform(X_train).toarray()

p =X_train.dtypes
dataset[dataset[1]]

mylist = list(X_train.select_dtypes(include=['object']).columns)
index
x = -1
for i in mylist:
    x+=1
    if()

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:54:53 2018

@author: dhruv
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv',low_memory=False)
dataset1= pd.read_csv('test.csv',low_memory=False)
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
y_train = dataset.iloc[:,80:81].values

X_test= dataset1.iloc[:,0:80]


X_train = X_train.drop(['Alley', 'MasVnrType','BsmtQual','BsmtCond','FireplaceQu','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageFinish','GarageQual','GarageCond','Fence','MiscFeature'],axis=1)
X_train = X_train

X_test = X_test.drop(['Alley', 'MasVnrType','BsmtQual','BsmtCond','FireplaceQu','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageFinish','GarageQual','GarageCond','Fence','MiscFeature'],axis=1)
X_test = X_test

op= X_train.isnull().sum()
no =[]
jp =X_test.isnull().sum()



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X_train[:, [3,24,49]])
X_train[:, [3,24,49]] = imputer.transform(X_train[:,[3,24,49]])

from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer1 = imputer1.fit(X_test[:, [3,24,49]])
X_test[:, [3,24,49]] = imputer1.transform(X_test[:,[3,24,49]])



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

li = list([2,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,25,26,27,32,33,34,45,47,52,62,63])
for i in li:
    labelencoder_X = LabelEncoder()
    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])
    
for j in li:
    labelencoder_X1 = LabelEncoder()
    X_test[:,j] = labelencoder_X1.fit_transform(X_test[:,j])
    
    
onehotencoder1 = OneHotEncoder(categorical_features=[0])
X_train = onehotencoder1.fit_transform(X_train).toarray()

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

#trainung our model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =300 , random_state=0)
regressor.fit(X_train,y_train)








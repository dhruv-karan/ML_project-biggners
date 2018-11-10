# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:14:26 2018

@author: dhruv
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('weight-height.csv')

dataset.info()
dataset.describe()

X = dataset.iloc[:,1:3].values
y = dataset.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y[:, 0] = labelencoder.fit_transform(y[:, 0])

#making traing and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)



# question1: pred male or female on given data set.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred  = y_pred>0.5
j=0
for i in y_pred:
    if(i==True):
        y_pred[j] = 1
        j =j+1
    y_pred[j]=0
    j = j+1
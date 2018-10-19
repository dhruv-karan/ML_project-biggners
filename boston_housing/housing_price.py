
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('boston_data.csv')
X = dataset.iloc[: ,0:14 ].values
y = dataset.iloc[:,13].values

#test and trainging sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#making model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#ploting graph
plt.scatter(X_train,y_train,color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')

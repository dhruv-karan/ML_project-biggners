
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
#plt.scatter(X_train,y_train,color='red')
#plt.plot(X_test, regressor.predict(X_test), color='blue')

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((404,1)).astype(int),values=X,axis=1)

X_opt = X[: , [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,1,2,3,4,5,6,7,8,9,10,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,1,2,3,4,5,6,7,8,9,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[: , [0,1,2,3,4,5,6,7,8,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,1,2,3,4,5,7,8,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,1,2,3,5,7,8,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0,1,2,3,7,8,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,1,2,3,7,12,13,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[: , [0,1,2,3,7,12,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,1,3,7,12,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3,7,12,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,7,12,14]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# hrer we can we that 7 12 14 is the variable with most sergenficance in pref=dicting 
#house prices
X_last  = dataset.iloc[:,[7,12,13]].values

from sklearn.cross_validation import train_test_split
X_last_train,X_last_test,y_last_train,y_last_test = train_test_split(X_last,y,test_size=0.2,random_state=0)

#making model
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_last_train,y_last_train)
value  = np.array([[20,4,10]])

y_pred1 = regressor1.predict(value)





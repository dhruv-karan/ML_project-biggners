# importind neccesary library
import pandas as pd
import numpy as np

#importing sata set
dataset = pd.read_csv('winequality-red.csv')
X= dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11:12].values

#making traing and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# pricting from regression model
y_pred =  regressor.predict(X_test)

#finding the most signficant factor

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((1599,1)).astype(int),values=X,axis=1)

X_opt = X[: , [0,1,2,3,4,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,2,3,4,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,2,3,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,2,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# predicting with this 

X_sign = dataset.iloc[:,[2,5,6,7,9,10,11]].values

from sklearn.cross_validation import train_test_split
X_sign_train, X_sign_test, y_train, y_test = train_test_split(X_sign, y, test_size=0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_sign_train, y_train)

#best prediction 
y_pred_best =  regressor1.predict(X_sign_test)


# feature 2,5,6,7,9,10,11 are the most significatnt factor in classifing wine quality 
# 





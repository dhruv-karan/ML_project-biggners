# importind neccesary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing sata set
dataset = pd.read_csv('winequality-red.csv')
# higher level data analysis

dataset.info()
dataset.head()
dataset.describe()

#making varaiable
X = dataset.drop(['quality'], axis = 1)
y = dataset['quality']

#making traing and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#adding 
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

#2,5,6,7,9,10,11 are the most significat independent varaible
# using 2 parameter
sns.barplot(x='quality', y='volatile acidity',data=dataset)

# using 5 th parameter
sns.barplot(x = 'quality', y='chlorides', data = dataset)

#6
sns.barplot(x = 'quality', y='free sulfur dioxide', data = dataset)
#7
sns.barplot(x = 'quality', y='total sulfur dioxide', data = dataset)
#9
sns.barplot(x = 'quality', y='pH', data = dataset)

#10
sns.barplot(x = 'quality', y='sulphates', data = dataset)
#11
sns.barplot(x = 'quality', y='alcohol', data = dataset)

# Encoding our dependent variable:Quality column
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling to X_train and X_test to classify better.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#fitting dataset in claasfier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the Test Set
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt='2.0f')

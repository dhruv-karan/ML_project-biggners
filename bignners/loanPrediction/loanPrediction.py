#importing libraray
import pandas as pd
import numpy as np
#import matplot.lib as plt

dataset  = pd.read_csv('test.csv')
X = dataset.iloc[:,3:10].values
y = dataset.iloc[:,10].values
#making dummy varaiable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#finding the misising values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:, 3])
X[:, 3] = imputer.transform(X[:,3])

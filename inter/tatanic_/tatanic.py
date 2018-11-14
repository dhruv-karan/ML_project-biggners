import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv('test.csv')
dataset3 = pd.read_csv('gender_submission.csv')
dataset.isnull().sum()
dataset1.isnull().sum()



dataset.dtypes

dataset.shape
dataset.head(30)
dataset.describe()

dataset.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

X_train= dataset.iloc[:,[2,4,5,6,7,9]].values
y_train= dataset.iloc[:,1:2].values

X_test = dataset1.iloc[:,[1,3,4,5,6,8]].values
y_test = dataset3.iloc[:,1:2].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:,2:3])

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X_test[:, [2,5]])
X_test[:, [2,5]] = imputer.transform(X_test[:,[2,5]])




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:,1] = labelencoder_X.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X1 = LabelEncoder()
X_test[:,1] = labelencoder_X1.fit_transform(X_test[:, 1])
onehotencoder1 = OneHotEncoder(categorical_features=[0])
X_test = onehotencoder1.fit_transform(X_test).toarray()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)



from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
        
y_pred = regressor.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# accuracy of 97.84%



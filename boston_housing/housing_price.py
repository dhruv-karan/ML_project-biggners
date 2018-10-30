
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('boston_data.csv')
X = dataset.iloc[: ,0:14 ].values
y = dataset.iloc[:,13:14].values

dataset.info()
dataset.head()
dataset.describe()

#test and trainging sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#fearure scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)


y_predict = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))



#print("lets predict price of u r dream house")
#print("enter the following values ")
#X_input=[12]
#X_input[0] = input("crim per capita crime rate by town.")
#X_input[1] = input("indus proportion of non-retail business acres per town..")
#X_input[2] = input("chas Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).")
#X_input[3] = input("nox nitrogen oxides concentration (parts per 10 million)..")
#X_input[4] = input("rm average number of rooms per dwelling.")
#X_input[5] = input("age proportion of owner-occupied units built prior to 1940.")
#X_input[6] = input("dis weighted mean of distances to five Boston employment centres.")
#X_input[7] = input("rad index of accessibility to radial highways.")
#X_input[8] = input("tax full-value property-tax rate per $10,000.")
#X_input[9] = input("ptratio pupil-teacher ratio by town.")
#X_input[10] = input("black 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.")
#X_input[11] = input("lstat lower status of the population (percent).")
#X_input[12] = input("medv median value of owner-occupied homes in $1000s."
    
    
    

#print(sc_y.inverse_transform(regressor.predict(sc_X.transform([X_input]))))
















#ploting graph
#plt.scatter(X_train,y_train,color='red')


#importing libraray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset  = pd.read_csv('Iris_Data.csv')
X = dataset.iloc[:,0:5].values

# making category and making dummy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,4] = labelencoder_X.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# lets remove dummy varaible trap
X = X[:, 1:]

# finding optimal numbe rof cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,max_iter=300,n_init=10,random_state=0)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11),wcss)

#applying kmeans on datet
Kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=100,n_init=6,random_state=0)
y_kmeans =Kmeans.fit_predict(X)

#vsiualise the Cluster0

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='target')

plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('cluster of clients')
plt.xlabel('annnual oncole')
plt.ylabel('spending score')
plt.legend()
plt.show()
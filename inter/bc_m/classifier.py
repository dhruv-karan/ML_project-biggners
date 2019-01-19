# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
df.columns
df.head(5)
df.describe

X = df.iloc[:,2:].values
y = df.iloc[:,1].values

le = LabelEncoder()
y = le.fit_transform(y)

le.transform(['M','B'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============== pipelining ===============

from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition import PCA
from  sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),('pca',PCA(n_components =2)),
                    ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train,y_train)
print(pipe_lr.score(X_test,y_test ))
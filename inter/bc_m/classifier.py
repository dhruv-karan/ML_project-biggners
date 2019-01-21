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

#================= Validation ==================
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y = y_train,n_folds=10,random_state=1)

scores = []

for k,(train,test) in enumerate(kfold):
    pipe_lr.fit(X_train[train],y_train[train])
    score = pipe_lr.score(X_train[test],y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,np.bincount(y_train[train]), score))

np.mean(scores)
np.std(scores)

#=============== visualing accuracy curve===============
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
pip_lr = Pipeline([
        ('scl',StandardScaler()),
        ('clf', LogisticRegression(penalty='12',random_state=0))])

train_sizes, train_scores,test_scores=\
    learning_curve(estimator=pipe_lr,
                   X=X_train,
                   y= y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10,
                   n_jobs=1)

train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5, label='TA')

plt.fill_between(train_sizes,train_mean+train_std,
                 train_mean - train_std,
                 alpha=0.15,color='blue')

plt.plot(train_sizes,test_mean,
         color='green',linestyle='--',
         marker='s',markersize=5,
         label='VA')
plt.fill_between(train_sizes,
                 test_mean+test_std,
                 test_mean-test_std,
                 alpha=0.15,color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()

# ================== Addressing overfitting and underfitting with validation curves============

from sklearn.learning_curve import validation_curve 
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train, 
        param_name='clf__C',
        param_range=param_range,
        cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
          color='blue', marker='o',
          markersize=5,
          label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                  train_mean - train_std, alpha=0.15,
                  color='blue') 

plt.plot(param_range, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy') 
plt.fill_between(param_range,test_mean + test_std, test_mean - test_std,alpha=0.15, color='green') 


plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()








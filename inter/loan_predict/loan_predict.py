# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Training data("for data related plz refer kaggle this competion 
# link ="https://www.kaggle.com/c/home-credit-default-risk"
# " )
df_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ', df_train.shape)
df_train.head()

df_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', df_test.shape)
df_test.head()

df_train['TARGET'].value_counts()
df_train['TARGET'].astype(int).plot.hist()

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# Number of each type of column
df_train.dtypes.value_counts()

# Number of unique classes in each object column
df_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# Create a label encoder object
le = LabelEncoder()
le_count = 0
for col in df_train:
    if(df_train[col].dtype =='object'):
        if len((df_train[col].unique())) <=2:
            le.fit(df_train[col])
            df_train[col] = le.transform(df_train[col])
            df_test[col] = le.transform(df_test[col])
            
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
#     if(col.dtype() == 'object'):
#         print(col)

# one-hot encoding of categorical variables
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)

##========================Aligning Training and Testing Data ==================
train_labels = df_train['TARGET']
# print(train_labels)

# # Align the training and testing data, keep only columns present in both dataframes
df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

# # # Add the target back in
df_train['TARGET'] = train_labels

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)

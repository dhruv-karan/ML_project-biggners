#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('Blackfriday.csv')

df_train.columns
l =df_train.isnull().sum()

df_train['Purchase'].describe()

sns.distplot(df_train['Purchase'])

#skewness and kurtosis
print("Skewness: %f" % df_train['Purchase'].skew())
print("Kurtosis: %f" % df_train['Purchase'].kurt())

#scatter plot grlivarea/saleprice
var = 'City_Category'
data = pd.concat([df_train['Purchase'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='Purchase', ylim=(0,24000))

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

## ===== terminating it again till got another method


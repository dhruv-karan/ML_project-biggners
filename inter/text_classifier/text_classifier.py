# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:58:20 2018

@author: dhruv
"""

import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
#importing datset

reviews = load_files('class/')
X, y = reviews.data, reviews.target

# storing file in pickel (for fast processing)
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle', 'wb') as f:
    pickle.dump(y,f)
# unpickling the file
with open('X.pickle','rb') as f:
    X= pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)
    
# filtering the data set 

corpus = []
for i in range(0, 2000):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)   
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from urllib.request import urlopen
import urllib.request
import csv
from bs4 import BeautifulSoup
from time import sleep
import re
import nltk
df  = pd.read_csv('cik_list.csv')
valid_df = df.iloc[:152,:].values
ur = valid_df[0][5]

for i in range(152):
    valid_df[i][5]="https://www.sec.gov/Archives/"+valid_df[i][5]
    print(valid_df[0][5])

full_data=[]

for i in range(152):
    url=valid_df[i][5]
    #print(url)
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,'html.parser')
    name_box = soup.find
    full_data.append(name_box) 


bag_sentence = []

for file in full_data:
    if(type(file) == 'str'or'object'):
        print(file)
        sentence = nltk.sent_tokenize(file)
        bag_sentence.append(sentence)
    else:
        pass

one = full_data[0]

nltk.sent_tokenize(one)


nam =  'my name.is dhruv karan'
sentences = nltk.sent_tokenize(nam)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]


j=nam.str.split(pat = '.')

type(one)















































with open('myfile1.txt', 'w+') as f:

    the_text = str(name_box)
    file = f.write(the_text)

with open('index1.csv', 'a+') as f:
    mydoc = csv.writer(f)
    for i in the_text.split('\n'):
        mydoc.writerow([i])


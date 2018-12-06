# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from urllib.request import urlopen
import urllib.request
import csv
from bs4 import BeautifulSoup
from time import sleep
import re

df  = pd.read_csv('cik_list.csv')
valid_df = df.iloc[:152,:].values



for i in range(152):
    valid_df[i][5]="https://www.sec.gov/Archives/"+valid_df[i][5]

full_data=[]


for i in range(5):
    url=valid_df[i][5]
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,'html.parser')
    name_box = str(soup.find)
    full_data.append(name_box)

bag_sentence = []

#for file in full_data:
 #       sentence = nltk.sent_tokenize(file)
  #      bag_sentence.append(sentence)
j=0
i =0
for j in range(len(bag_sentence)):
    print(j)
    for i in range(len(bag_sentence[j])):
        bag_sentence[j][i] = re.sub(r'\[[0-9]*\]',' ',bag_sentence[j][i])
        bag_sentence[j][i] = bag_sentence[j][i].lower()
        bag_sentence[j][i] = re.sub(r'\d',' ',bag_sentence[j][i])
        bag_sentence[j][i] = re.sub(r'^b\s+', '', bag_sentence[j][i])
        bag_sentence[j][i] = re.sub(r'\s+[a-z]\s+', ' ',bag_sentence[j][i])
        bag_sentence[j][i] = re.sub(r'[@#.,/[/]=/</>/-:$/(/)]',' ', bag_sentence[j][i])
        bag_sentence[j][i] = re.sub(r'[^\w]', ' ', bag_sentence[j][i])
        bag_sentence[j][i] = re.sub(r'\s+',' ',bag_sentence[j][i])
    
    
bag_words = []

#for sentences in bag_sentence:
 #   words = [nltk.word_tokenize(sentence) for sentence in sentences]
  #  bag_words.append(words)


bag_stopwords =[]

#for p in range(len(bag_words)):
 #   for b in range(len(bag_words[p])):
  #      bag_words[p][b] = [word for word in bag_words[p][b] if word not in stopwords.words('english')]




























with open('myfile1.txt', 'w+') as f:

    the_text = str(name_box)
    file = f.write(the_text)

with open('index1.csv', 'a+') as f:
    mydoc = csv.writer(f)
    for i in the_text.split('\n'):
        mydoc.writerow([i])


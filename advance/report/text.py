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
df1 = pd.read_csv('resources/sw/StopWords_Auditor.txt', header=None )
#df2 = pd.read_csv('resources/sw/StopWords_Currencies.txt', header=None,delimiter="|")
df3 = pd.read_csv('resources/sw/StopWords_DatesandNumbers.txt', header=None )
df4 = pd.read_csv('resources/sw/StopWords_Generic.txt', header=None )
df5 = pd.read_csv('resources/sw/StopWords_GenericLong.txt', header=None )
df6 = pd.read_csv('resources/sw/StopWords_Geographic.txt', header=None )
df7 = pd.read_csv('resources/sw/StopWords_Names.txt', header=None )

resource = []
stopwords =[]
resource.append(df1)
resource.append(df3)
resource.append(df4)
resource.append(df5)
resource.append(df6)
resource.append(df7)

for i in resource:
    li = list(i.iloc[:,:].values)
    for ln in li:
        inter = ln.tolist()
        stopwords.append(inter[0])
        #print(type(ln))
lower_stopwords =[]
for stop in stopwords:
    if(type(stop)==str):
        s = stop
        lower_stopwords.append(s.lower())


for i in range(152):
    valid_df[i][5]="https://www.sec.gov/Archives/"+valid_df[i][5]

full_data=[]


for i in range(3):
    url=valid_df[i][5]
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,'html.parser')
    name_box = str(soup.find)
    full_data.append(name_box)

bag_sentence = []

for file in full_data:
    sentence = nltk.sent_tokenize(file)
    bag_sentence.append(sentence)
j=0
i =0
for j in range(len(bag_sentence)):
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

for sentences in bag_sentence:
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    bag_words.append(words)


bag_stopwords =[]

for p in range(len(bag_words)):
    for b in range(len(bag_words[p])):
        for st in lower_stopwords:
            for words in bag_words[p][b]:
                if(words ==st):
                    if(st):
                        print(words)
                        bag_words[p][b].remove(words)
                    else:
                        pass
                else:
                    pass

































with open('myfile1.txt', 'w+') as f:

    the_text = str(name_box)
    file = f.write(the_text)

with open('index1.csv', 'a+') as f:
    mydoc = csv.writer(f)
    for i in the_text.split('\n'):
        mydoc.writerow([i])


# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import urllib.request
import csv
from bs4 import BeautifulSoup
import re
import nltk

#===========================================imported data ===================================
df  = pd.read_csv('cik_list.csv')
valid_df = df.iloc[:152,:].values

#================================================== imported stopwords  stop words=========================
df1 = pd.read_csv('resources/sw/StopWords_Auditor.txt', header=None )
df3 = pd.read_csv('resources/sw/StopWords_DatesandNumbers.txt', header=None )
df4 = pd.read_csv('resources/sw/StopWords_Generic.txt', header=None )
df5 = pd.read_csv('resources/sw/StopWords_GenericLong.txt', header=None )
df6 = pd.read_csv('resources/sw/StopWords_Geographic.txt', header=None )
df7 = pd.read_csv('resources/sw/StopWords_Names.txt', header=None )

#=============================== importing master words==================================================
df8 = pd.read_csv('resources/word/LM_Negative.txt',header=None,delimiter="\ ")
df9 = pd.read_csv('resources/word/LM_Positive.txt',header=None,delimiter="\ ")

#============================================ making list of negative words =================================
neg = list(df8.iloc[:,:].values)
type(neg[1].tolist())

negative = []
for ne in neg:
    negi = ne.tolist()
    for xx in range(len(negi)):
       neg_word = negi[xx]
       if(type(neg_word) ==str):
           negative.append(neg_word.lower())
       else:
            pass

#============================================making list of positive words =========================
pos = list(df9.iloc[:,:].values)
type(pos[1].tolist())

positive = []
for po in pos:
    posi = po.tolist()
    for xxx in range(len(posi)):
       pos_word = posi[xxx]
       if(type(pos_word) ==str):
           positive.append(pos_word.lower())
       else:
            pass
#======================================   made stopwords list ===============================================        
resource = []                       
resource.append(df1)
resource.append(df3)
resource.append(df4)
resource.append(df5)
resource.append(df6)
resource.append(df7)

stopwords =[]

for i in resource:
    li = list(i.iloc[:,:].values)
    for ln in li:
        inter = ln.tolist()
        lo = inter[0]
        if(type(lo)==str):            
            stopwords.append(lo.lower())

# ======================================== mining data from links =====================================
for i in range(152):
    valid_df[i][5]="https://www.sec.gov/Archives/"+valid_df[i][5]

full_data=[]
for i in range(3):
    url=valid_df[i][5]
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,'html.parser')
    name_box = str(soup.find)
    full_data.append(name_box)


#=========================== ============= tokenising sentence ==================
bag_sentence = []
for file in full_data:
    sentence = nltk.sent_tokenize(file)
    bag_sentence.append(sentence)
    
#  ========================================    preprocessing ====================
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
    

# ====================================== tokensizing words nltk ====================
bag_words = []
for sentences in bag_sentence:
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    bag_words.append(words)

#=========================    removing stop words ======================
for p in range(len(bag_words)):
    for b in range(len(bag_words[p])):
        for st in stopwords:
            for words in bag_words[p][b]:
                if(words == st):
                    if(st):
                        bag_words[p][b].remove(words)
                    else:
                        pass
                else:
                    pass
  
pos_list = [[]]
for aa in range(len(bag_words)):
    for bb in range(len(bag_words[aa])):
        count = 0
        for posw in positive:
            for cc in bag_words[aa][bb]:
                if(cc == posw):
                    count +=count
                    



                
                
     




           
                
                
                
                
with open('myfile1.txt', 'w+') as f:

    the_text = str(name_box)
    file = f.write(the_text)

with open('index1.csv', 'a+') as f:
    mydoc = csv.writer(f)
    for i in the_text.split('\n'):
        mydoc.writerow([i])


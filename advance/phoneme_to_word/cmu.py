import pandas as pd
import numpy as np
# #read in data
# prime_dictionary = open('cmudict.dict', 'r')
# punct_dictionary = open('cmudict.vp', 'r')

# word = []
# pronunciation = []
# def compile(dictionary):
#     with dictionary as f:
#         phonics = [line.rstrip('\n') for line in f]

#     for x in phonics:
#         x = x.split(' ')
#         word.append(x[0])
#         p = ' '.join(x[1:])
#         pronunciation.append(p)
# compile(prime_dictionary)
# # comment out the following line if you do not want punctuation pronunciations in the DataFrame
# compile(punct_dictionary)

# # make the dataset   
# result = pd.DataFrame({"word": word})
# result['pronunciation'] = pronunciation
# result[:20]

# result.to_csv("./cmudict.csv", index=True, header=True)

# =============================  UNCOMMENT THE LINES FOR CONVERTING DOWNLOADED FILLES INTO CSV FILE=========================

df = pd.read_csv('cmudict.csv')

df.head(2)

#======================= making function for makinng one hot vector for words ==========================

word_phone = [i.split() for i in df['pronunciation']]

words_orginal = [i for i in df['word']]

unique = []
for i in df['pronunciation']:
    word = i.split()
    for j in range(len(word)):
        if word[j] not in unique:
            unique.append(word[j])
        else: pass   
del unique[69:]


def one_hot(phone_list):
    hot_vec = np.zeros((len(unique)))
    for word in phone_list:
        pos = unique.index(word)
        hot_vec[pos] = 1
    return hot_vec


def one_hot_word(word):
    hot_vec = np.zeros((len(df['word'])))
    pos = words_orginal.index(word)
    hot_vec[pos] = 1
    return hot_vec

    

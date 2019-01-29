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

df.head(20)



# ====================   making char embbeding  dict ====================

word_phone = [i.split() for i in df['pronunciation']]

unique = []
for i in df['pronunciation']:
    word = i.split()
    for j in range(len(word)):
        if word[j] not in unique:
            unique.append(word[j])
        else: pass   
del unique[69:]



def one_hot(phone_list):
    hot_vec_list =[]
    for word in phone_list:
        hot_vec = np.zeros((len(unique)))
        pos = phone_list.index(word)
        hot_vec[pos] = 1
        hot_vec_list.append(hot_vec)
    return hot_vec_list


one_hot(word_phone[0])






















import string

START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'


def char_list():
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    return [''] + allowed_symbols + uppercase_letters


def phone_list():
    phone_list = [START_PHONE_SYM, END_PHONE_SYM]
    for line in df['pronunciation']:
        phone_list.append(line.strip())
    return [''] + phone_list


def id_mappings_from_list(str_list):
    str_to_id = {s: i for i, s in enumerate(str_list)}
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str


# Create character to ID mappings
char_to_id, id_to_char = id_mappings_from_list(char_list())

# Load phonetic symbols and create ID mappings
phone_to_id, id_to_phone = id_mappings_from_list(phone_list())

# Example:
print('Char to id mapping: \n', char_to_id)


#============================== converting chacter into one hot vector ======================
CHAR_TOKEN_COUNT = len(char_to_id)
PHONE_TOKEN_COUNT = len(phone_to_id)


def char_to_1_hot(char):
    char_id = char_to_id[char]
    hot_vec = np.zeros((CHAR_TOKEN_COUNT))
    hot_vec[char_id] = 1.
    return hot_vec


def phone_to_1_hot(phone):
    phone_id = phone_to_id[phone]
    print(phone_id)
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1.
    return hot_vec

# Example:
print('"A" is represented by:\n', char_to_1_hot('B'), '\n-----')
print('"AH0" is represented by:\n', phone_to_1_hot('AA0 B AA0 T IY0 EH1 L OW0'))

df.columns
phonetic_dict ={w: p for w,p in zip(df['word'],df['pronunciation'])}


MAX_CHAR_SEQ_LEN = max([len(str(word)) for word, _ in phonetic_dict.items()])
MAX_PHONE_SEQ_LEN = max([max([len(pron.split()) for pron in pronuns])
                         for _, pronuns in phonetic_dict.items()]) + 2  # + 2 to account for the start & end tokens we need to add


def dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []
    
    for word, pronuns in phonetic_dict.items():
        word_matrix = np.zeros((MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
        for t, char in enumerate(word):
            word_matrix[t, :] = char_to_1_hot(char)
        for pronun in pronuns:
            pronun_matrix = np.zeros((MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
            phones = [START_PHONE_SYM] + pronun.split() + [END_PHONE_SYM]
            for t, phone in enumerate(phones):
                pronun_matrix[t,:] = phone_to_1_hot(phone)
                
            char_seqs.append(word_matrix)
            phone_seqs.append(pronun_matrix)
    
    return np.array(char_seqs), np.array(phone_seqs)
            

char_seq_matrix, phone_seq_matrix = dataset_to_1_hot_tensors()
print('Word Matrix Shape: ', char_seq_matrix.shape)
print('Pronunciation Matrix Shape: ', phone_seq_matrix.shape)


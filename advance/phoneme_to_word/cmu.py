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
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1.
    return hot_vec

# Example:
print('"A" is represented by:\n', char_to_1_hot('A'), '\n-----')
print('"AH0" is represented by:\n', phone_to_1_hot('AH0'))



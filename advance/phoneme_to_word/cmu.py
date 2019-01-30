import re
import os
import random
import numpy as np



IS_KAGGLE = True 

CMU_DICT_PATH = os.path.join('cmudict-0.7b')
CMU_SYMBOLS_PATH = os.path.join('cmudict.symbols')

# Skip words with numbers or symbols
ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"

# Only 3 words are longer than 20 chars
# Setting a limit now simplifies training our model later
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2



def load_clean_phonetic_dictionary():

    def is_alternate_pho_spelling(word):
        # No word has > 9 alternate pronounciations so this is safe
        return word[-1] == ')' and word[-3] == '(' and word[-2].isdigit() 

    def should_skip(word):
        if not word[0].isalpha():  # skip symbols
            return True
        if word[-1] == '.':  # skip abbreviations
            return True
        if re.search(ILLEGAL_CHAR_REGEX, word):
            return True
        if len(word) > MAX_DICT_WORD_LEN:
            return True
        if len(word) < MIN_DICT_WORD_LEN:
            return True
        return False

    phonetic_dict = {}
    with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict:
        for line in cmu_dict:

            # Skip commented lines
            if line[0:3] == ';;;':
                continue

            word, phonetic = line.strip().split('  ')

            # Alternate pronounciations are formatted: "WORD(#)  F AH0 N EH1 T IH0 K"
            # We don't want to the "(#)" considered as part of the word
            if is_alternate_pho_spelling(word):
                word = word[:word.find('(')]

            if should_skip(word):
                continue

            if word not in phonetic_dict:
                phonetic_dict[word] = []
            phonetic_dict[word].append(phonetic)

    if IS_KAGGLE: # limit dataset to 5,000 words
        phonetic_dict = {key:phonetic_dict[key] 
                         for key in random.sample(list(phonetic_dict.keys()), 5000)}
    return phonetic_dict

phonetic_dict = load_clean_phonetic_dictionary()
example_count = np.sum([len(prons) for _, prons in phonetic_dict.items()])






print("\n".join([k+' --> '+phonetic_dict[k][0] for k in random.sample(list(phonetic_dict.keys()), 10)]))
print('\nAfter cleaning, the dictionary contains %s words and %s pronunciations (%s are alternate pronunciations).' % 
      (len(phonetic_dict), example_count, (example_count-len(phonetic_dict))))



import string

START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'


def char_list():
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    return [''] + allowed_symbols + uppercase_letters


def phone_list():
    phone_list = [START_PHONE_SYM, END_PHONE_SYM]
    with open(CMU_SYMBOLS_PATH) as file:
        for line in file: 
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


MAX_CHAR_SEQ_LEN = max([len(word) for word, _ in phonetic_dict.items()])
MAX_PHONE_SEQ_LEN = max([max([len(pron.split()) for pron in pronuns]) 
                         for _, pronuns in phonetic_dict.items()]
                       ) + 2  # + 2 to account for the start & end tokens we need to add


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




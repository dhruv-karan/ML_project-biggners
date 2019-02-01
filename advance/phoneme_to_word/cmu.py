import re
import os
import random
import numpy as np


import pandas as pd

df= pd.read_csv('phoneme_to_word/cmudict.csv')

df.isnull().sum()
df.dropna(inplace=True)
df.columns
df.head(3)

word = df['word']
pronunciation = df['pronunciation']

import string

START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'


def char_list():
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    return [''] + allowed_symbols + uppercase_letters

def phone_list():
    phone_list = [START_PHONE_SYM, END_PHONE_SYM]
    for i in pronunciation:
        i = i.split(' ')
        for n in i:
            if n not in phone_list:
                phone_list.append(n)
            else:
                pass
    return [''] + phone_list


def id_mappings_from_list(str_list):
    str_to_id = {s: i for i, s in enumerate(str_list)}
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str

# Create character to ID mappings
cha_to_id, id_to_cha = id_mappings_from_list(char_list())

# Load phonetic symbols and create ID mappings
phoe_to_id, id_to_phoe = id_mappings_from_list(phone_list())

# Example:
print('Char to id mapping: \n',phoe_to_id)



CHAR_TOKEN_COUNT = len(cha_to_id)
PHONE_TOKEN_COUNT = len(phoe_to_id)


def char_to_1_hot(char):
    char_id = cha_to_id[char.upper()]
    hot_vec = np.zeros((CHAR_TOKEN_COUNT))
    hot_vec[char_id] = 1
    return hot_vec


def phone_to_1_hot(phone):
    phone_id = phoe_to_id[phone]
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1
    return hot_vec

# Example:
print('"A" is represented by:\n', char_to_1_hot('A'), '\n-----')
print('"AH0" is represented by:\n', phone_to_1_hot('AH0'))


MAX_CHAR_SEQ_LEN = max([len(w) for w in word])
MAX_PHONE_SEQ_LEN = max([max([len(pron) for pron in pronunciation])]) +2 # + 2 to account for the start & end tokens we need to add


def dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []
    
    for w in word:
        word_matrix = np.zeros((MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
        for t,char in enumerate(w):
            if char not in cha_to_id: pass
            else: word_matrix[t, :] = char_to_1_hot(char)
        char_seqs.append(word_matrix)
        
    for pronun in pronunciation:
        pronun_matrix = np.zeros((MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
        phones = [START_PHONE_SYM] + pronun.split() + [END_PHONE_SYM]
        for t, phone in enumerate(phones):
                pronun_matrix[t,:] = phone_to_1_hot(phone)
        phone_seqs.append(pronun_matrix)
    
    return np.array(char_seqs),np.array(phone_seqs)
            
char_seq_matrix, phone_seq_matrix = dataset_to_1_hot_tensors()
print('Word Matrix Shape: ', char_seq_matrix.shape)
print('Pronunciation Matrix Shape: ', phone_seq_matrix.shape)


phone_seq_matrix_decoder_output = np.pad(phone_seq_matrix,((0,0),(0,1),(0,0)), mode='constant')[:,1:,:]










from keras.models import Model
from keras.layers import Input, LSTM, Dense

def baseline_model(hidden_nodes = 256):
    
    # Shared Components - Encoder
    char_inputs = Input(shape=(None, CHAR_TOKEN_COUNT))
    encoder = LSTM(hidden_nodes, return_state=True)
    
    # Shared Components - Decoder
    phone_inputs = Input(shape=(None, PHONE_TOKEN_COUNT))
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True)
    decoder_dense = Dense(PHONE_TOKEN_COUNT, activation='softmax')
    
    # Training Model
    _, state_h, state_c = encoder(char_inputs) # notice encoder outputs are ignored
    encoder_states = [state_h, state_c]
    decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
    phone_prediction = decoder_dense(decoder_outputs)

    training_model = Model([char_inputs, phone_inputs], phone_prediction)
    
    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)
    
    # Testing Model - Decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    phone_prediction = decoder_dense(decoder_outputs)
    
    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)
    
    return training_model, testing_encoder_model, testing_decoder_model


from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
    
(char_input_train, char_input_test, 
 phone_input_train, phone_input_test, 
 phone_output_train, phone_output_test) = train_test_split(
    char_seq_matrix, phone_seq_matrix, phone_seq_matrix_decoder_output, 
    test_size=TEST_SIZE, random_state=42)

TEST_EXAMPLE_COUNT = char_input_test.shape[0]

from keras.callbacks import ModelCheckpoint, EarlyStopping

def train(model, weights_path, encoder_input, decoder_input, decoder_output):
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss',patience=3)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=256,
          epochs=100,
          validation_split=0.2, # Keras will automatically create a validation set for us
          callbacks=[checkpointer, stopper])
    
BASELINE_MODEL_WEIGHTS = os.path.join(
    '../input', 'predicting-english-pronunciations-model-weights', 'baseline_model_weights.hdf5')
training_model, testing_encoder_model, testing_decoder_model = baseline_model()
if not IS_KAGGLE:
    train(training_model, BASELINE_MODEL_WEIGHTS, char_input_train, phone_input_train, phone_output_train)

def predict_baseline(input_char_seq, encoder, decoder):
    state_vectors = encoder.predict(input_char_seq) 
    
    prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
    prev_phone[0, 0, phone_to_id[START_PHONE_SYM]] = 1.
    
    end_found = False 
    pronunciation = '' 
    while not end_found:
        decoder_output, h, c = decoder.predict([prev_phone] + state_vectors)
        
        # Predict the phoneme with the highest probability
        predicted_phone_idx = np.argmax(decoder_output[0, -1, :])
        predicted_phone = id_to_phone[predicted_phone_idx]
        
        pronunciation += predicted_phone + ' '
        
        if predicted_phone == END_PHONE_SYM or len(pronunciation.split()) > MAX_PHONE_SEQ_LEN: 
            end_found = True
        
        # Setup inputs for next time step
        prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
        prev_phone[0, 0, predicted_phone_idx] = 1.
        state_vectors = [h, c]
        
    return pronunciation.strip()

def one_hot_matrix_to_word(char_seq):
    word = ''
    for char_vec in char_seq[0]:
        if np.count_nonzero(char_vec) == 0:
            break
        hot_bit_idx = np.argmax(char_vec)
        char = id_to_char[hot_bit_idx]
        word += char
    return word


# Some words have multiple correct pronunciations
# If a prediction matches any correct pronunciation, consider it correct.
def is_correct(word,test_pronunciation):
    correct_pronuns = phonetic_dict[word]
    for correct_pronun in correct_pronuns:
        if test_pronunciation == correct_pronun:
            return True
    return False


def sample_baseline_predictions(sample_count, word_decoder):
    sample_indices = random.sample(range(TEST_EXAMPLE_COUNT), sample_count)
    for example_idx in sample_indices:
        example_char_seq = char_input_test[example_idx:example_idx+1]
        predicted_pronun = predict_baseline(example_char_seq, testing_encoder_model, testing_decoder_model)
        example_word = word_decoder(example_char_seq)
        pred_is_correct = is_correct(example_word, predicted_pronun)
        print('✅ ' if pred_is_correct else '❌ ', example_word,'-->', predicted_pronun)


training_model.load_weights(BASELINE_MODEL_WEIGHTS)  # also loads weights for testing models
sample_baseline_predictions(10, one_hot_matrix_to_word)

import math


def get_change(m):
    d = m%10
    if d==0:
        return m/10
    m = math.floor(m/10)
    if d>5:
        m = m+ d-5
        return m
    d = d%5
    if d==0:
        m = m+1
        return m
    m = m+d
    return m

get_change(9)
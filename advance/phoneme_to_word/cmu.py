import pandas as pd

#read in data
prime_dictionary = open('cmudict.dict', 'r')
punct_dictionary = open('cmudict.vp', 'r')

word = []
pronunciation = []
def compile(dictionary):
    with dictionary as f:
        phonics = [line.rstrip('\n') for line in f]

    for x in phonics:
        x = x.split(' ')
        word.append(x[0])
        p = ' '.join(x[1:])
        pronunciation.append(p)
compile(prime_dictionary)
# comment out the following line if you do not want punctuation pronunciations in the DataFrame
compile(punct_dictionary)

# make the dataset   
result = pd.DataFrame({"word": word})
result['pronunciation'] = pronunciation
result[:20]

result.to_csv("./cmudict.csv", index=True, header=True)

df = pd.read_csv('cmudict.csv')
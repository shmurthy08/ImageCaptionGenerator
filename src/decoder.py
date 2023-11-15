import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.layers import LayerNormalization
import pickle
import os
import string


# Load GloVe embeddings
glove_embeddings = {}
with open('./glove_embeddings/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        glove_embeddings[word] = coefs
        

# Load captions.txt into mem
def load_doc(filename):
    # open file
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# extract descriptions
def load_description(doc):
    map = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        # first token = img id and rest is description
        img_id, img_dec = tokens[0], tokens[1:]
        img_id = img_id.split('.')[0]
        img_dec = ' '.join(img_dec)
        map[img_id] = list()
        map[img_id].append(img_dec)
    return map


def clean_descriptions(descriptions):
    tbl = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # conv to lower case; remove punc and numbers
            desc = [word.lower() for word in desc]
            desc = [w.translate(tbl) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
			# store as string
            desc_list[i] =  ' '.join(desc)


# convert to vocab of words
def vocab(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# save descriptions
def save_desc(desc, filename):
    lines = list()
    for key, desclist in desc.items():
        for desc in desclist:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
filename = captions_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'Flicker8k', 'captions.txt')
doc = load_doc(filename)
descriptions = load_description(doc)
print('Loaded: %d ' % len(descriptions))
clean_descriptions(descriptions)
vocab = vocab(descriptions)
print('Vocabulary Size: %d' % len(vocab))

save_desc(descriptions, 'descriptions.txt')


# # Read in the captions from Flicker8k

# captions = []
# captions_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'Flicker8k', 'captions.txt')
# with open(captions_dir, 'r') as file:
#     lines = file.readlines()
#     captions = [line.split(',', 1)[1].strip() for line in lines] # we want just the captions not the image it's associated with
    
# # Tokenize and preprocess captions
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(captions)
# sequences = tokenizer.texts_to_sequences(captions)
# max_len = max(len(seq) for seq in sequences)
# pad_seq = pad_sequences(sequences, maxlen=max_len, padding='post')

# embedding_dim = len(glove_embeddings['a'])
# vocab_size = len(tokenizer.word_index) + 1 


# embedding_matrix = np.zeros((vocab_size, embedding_dim))

# for word, i in tokenizer.word_index.items():
#     embedding_vec = glove_embeddings.get(word)
#     if embedding_vec is not None:
#         embedding_matrix[i] = embedding_vec
        
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=True))
# model.add(LSTM(100, return_sequences=True))
# model.add(LayerNormalization()) 
# model.add(LSTM(100)) 
# model.add(Dense(100, activation='relu'))
# model.add(Dense(vocab_size, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam')

# # Load image representations from pickle file
# with open('extracted_feats.pkl', 'rb') as file:
#     image_representations = pickle.load(file)


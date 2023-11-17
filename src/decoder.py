"""
This script defines a decoder model for an image caption generator using TensorFlow and Keras. It loads GloVe embeddings, preprocesses the captions, tokenizes them, and creates sequences of input image features and output captions for training the model. The decoder model is defined using an embedding layer, two LSTM layers, two layer normalization layers, and two dense layers. The model is compiled using categorical cross-entropy loss and the Adam optimizer, and is trained on the input image features and output captions. 
"""
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout, add
from tensorflow.keras.layers import LayerNormalization, BatchNormalization
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt


from nltk.translate.bleu_score import corpus_bleu



# Load image features
with open('extracted_feats.pkl', 'rb') as f:
    image_features = pickle.load(f)


# Load GloVe embeddings
glove_embeddings = {}
with open('./glove_embeddings/glove.6B.200d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        glove_embeddings[word] = coefs
        

# Create an empty dictionary to store image IDs and their corresponding captions
map = {}

# Open the captions.txt file and read its contents
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'Flicker8k', 'captions.txt')) as f:
    # Skip the first line (header)
    next(f)
    # Read the rest of the file
    captions_doc = f.read()

#process lines
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in map:
        map[image_id] = []
    # store the caption
    map[image_id].append(caption)
    
print(len(map))

##########################

# PREPROCESSING

def cleaning(map):
    """
    This function takes a dictionary of image names and their corresponding captions as input.
    It cleans the captions by converting them to lowercase, removing non-alphabetic characters,
    adding start and end tags to the captions, and deleting additional spaces.

    Args:
    - map: A dictionary of image names and their corresponding captions.

    Returns:
    - None. The function modifies the input dictionary in place.
    """

    # iterate over the dictionary items
    for key, caps in map.items():
        # iterate over the captions for each image
        for i in range(len(caps)):
            caption = caps[i]
            # convert the caption to lowercase
            caption = caption.lower()
            # remove non-alphabetic characters
            caption = [word for word in caption.split() if word.isalpha()]
            caption = ' '.join(caption)
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            caps[i] = caption
            
cleaning(map)
print(map['1000268201_693b08cb0e'])
# Create an empty list to store all captions
all_captions = []

# Pickle the map dictionary
with open('map.pkl', 'wb') as f:
    pickle.dump(map, f)

# Iterate over the dictionary items
for key in map: 
    # Iterate over the captions for each image
    for cap in map[key]:
        # Append the caption to the list of all captions
        all_captions.append(cap)
        
# print(len(all_captions)) uncomment for test purposes
 

#########################

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1


# uncomment below line for test reasons
# print(vocab_size) 

# get max length
max_length = max(len(caption.split()) for caption in all_captions)

print(max_length)

# TTS time - manual TTS for sequential reasons
img_ids = list(map.keys())


split = int(len(img_ids) * 0.80)
train = img_ids[:split]
test = img_ids[split:]


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size=128):
    """
    Generator function that yields a batch of data at a time.

    Args:
        data_keys (list): List of image IDs.
        mapping (dict): Dictionary of image IDs and their corresponding captions.
        features (dict): Dictionary of image IDs and their corresponding features.
        tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer object used to convert text to sequences.
        max_length (int): Maximum length of input sequence.
        vocab_size (int): Size of vocabulary.
        batch_size (int): Batch size for training.

    Yields:
        tuple: A tuple of numpy arrays containing input image features, input sequences, and output sequences.
    """
    X1, X2, y = [], [], []
    while True:
        for key, caps in mapping.items():
            feature = features[key][0]
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(y)
                        X1, X2, y = [], [], []

# Splitting data into train and test sets
train_cap = []
test_cap = []
for key, caps in map.items():
    if key in train:
        [train_cap.append(cap) for cap in caps]
    else:
        [test_cap.append(cap) for cap in caps]

# print(train_cap)
# print(test_cap)




# # Combining GloVe with our caption embeddings
# embedding_dim = len(glove_embeddings['a'])
# embed_matrix = np.zeros((vocab_size, embedding_dim))

# for word, index in tokenizer.word_index.items():
#     embedding_vec = glove_embeddings.get(word)
#     if embedding_vec is not None:
#         embed_matrix[index] = embedding_vec
        

# print("Image Features Shape: ", len(image_features))

# # Define encoder model
# inputs1 = Input(shape=(2048,))
# fe1 = BatchNormalization()(inputs1)
# fe2 = Dense(256, activation='relu')(fe1)

# # Seq feature layer
# inputs2 = Input(shape=(max_length,))
# se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
# se2 = LayerNormalization()(se1)
# se3 = LSTM(256, return_sequences=True)(se2)
# norm = LayerNormalization()(se3)
# se4 = LSTM(256)(norm)
# norm2 = LayerNormalization()(se4)

# # Decoder model
# decoder1 = add([fe2, norm2])
# norm3 = LayerNormalization()(decoder1)
# decoder2 = Dense(256, activation='relu')(norm3)
# outputs = Dense(vocab_size, activation='softmax')(decoder2)

# # Tie it together
# decoder_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
# decoder_model.summary()
# plot_model(decoder_model, to_file='complete_arch.png', show_shapes=True)

# # Compile the model
# decoder_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# steps = len(train_cap)//128
# epochs = 35 
# val_steps=len(test_cap)//128

# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}/{epochs}")
#     try:
#         generator = data_generator(train, map, image_features, tokenizer, max_length, vocab_size)
#         decoder_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#         decoder_model.save('model.h5')
#         print(f"Saved model on epoch: {epoch +1}")
#     except StopIteration:
#         break
#     print()  # Add a newline after each epoch


# print("Saved model to disk")
# with open('tokenizer.pkl', 'wb') as f:
#     pickle.dump(tokenizer, f)
# print("Saved tokenizer to disk")



model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
    
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



def pred_caption(model, image, tokenizer, max_length):
    input_txt = 'startseq'
    for i in range(max_length):
        #encode input
        seq = tokenizer.texts_to_sequences([input_txt])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        #predict next word
        y_hat = model.predict([image, seq], verbose=0)
        y_hat = np.argmax(y_hat)
        word = word_for_id(y_hat, tokenizer)
        if word is None:
            break
        
        
        
        input_txt += ' ' + word
        if word == 'endseq':
            break 
      
    return input_txt


# Let's use the test data and assess performance using BLEU Score
# for k in test:
#     #get caption
#     all_caps = map[k]
#     # predict caption
#     y_pred = pred_caption(m, image_features[k], tokenizer, max_length)
#     y_actual = [caption.split() for caption in all_caps]
#     # calc BLEU score
#     print(f"BLEU-1: {corpus_bleu(y_actual[0], y_pred, weights=(1.0,0,0))}")





def gener_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'Flicker8k', 'Images', image_name)
    img = Image.open(img_path)
    captions = map[image_id]
    for cap in captions:
        print(cap)
    y_pred = pred_caption(model, image_features[image_id], tokenizer, max_length)
    
    print(y_pred[9:-7])
    # plt.imshow(img)
    
    
gener_caption('3712008738_1e1fa728da.jpg')

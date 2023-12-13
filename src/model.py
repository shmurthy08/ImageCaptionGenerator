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
import random
# BLEU Score Import
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from tensorflow.keras.preprocessing import image as kimage
#Feature extraction



features = load_model('features.h5')
def extract_features(image_path):
    img = kimage.load_img(image_path, target_size=(224, 224))
    img_array = kimage.img_to_array(img)
    img_array = img_array/255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    extracted_features = features.predict(img_array)
    return extracted_features



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




# Combining GloVe with our caption embeddings
embedding_dim = len(glove_embeddings['a'])
embed_matrix = np.zeros((vocab_size, embedding_dim))

for word, index in tokenizer.word_index.items():
    embedding_vec = glove_embeddings.get(word)
    if embedding_vec is not None:
        embed_matrix[index] = embedding_vec
        

# Define encoder model
inputs1 = Input(shape=(2048,))
fe1 = BatchNormalization()(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Seq feature layer
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = LayerNormalization()(se1)
se3 = LSTM(256, return_sequences=True)(se2)
norm = LayerNormalization()(se3)
se4 = LSTM(256)(norm)
norm2 = LayerNormalization()(se4)

# Decoder model
decoder1 = add([fe2, norm2])
norm3 = LayerNormalization()(decoder1)
decoder2 = Dense(256, activation='relu')(norm3)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# Tie it together
decoder_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
decoder_model.summary()
plot_model(decoder_model, to_file='complete_arch.png', show_shapes=True)

# Compile the model
decoder_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


steps = len(train_cap)//128
epochs = 35 

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    try:
        generator = data_generator(train, map, image_features, tokenizer, max_length, vocab_size)
        decoder_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        decoder_model.save('model.h5')
        print(f"Saved model on epoch: {epoch +1}")
    except StopIteration:
        break
    print()  # Add a newline after each epoch


print("Saved model to disk")
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Saved tokenizer to disk")



model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
    
def word_for_id(integer, tokenizer):
    """
    Retrieves the word corresponding to the given integer index from the tokenizer's word index.

    Parameters:
    integer (int): The integer index of the word.
    tokenizer (Tokenizer): The tokenizer object containing the word index.

    Returns:
    str or None: The word corresponding to the given integer index, or None if not found.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



def pred_caption(model, image, tokenizer, max_length):
    """
    Generate a caption for an image using a given model, tokenizer, and maximum length.

    Parameters:
    - model: The image captioning model.
    - image: The input image.
    - tokenizer: The tokenizer used to encode the input text.
    - max_length: The maximum length of the generated caption.

    Returns:
    - The generated caption for the image.
    """
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

# pred caption for any image passed in; not in dataset
def pred_caption_any_img(model, img_path, tokenizer, max_length):
    """
    Generates a caption for an image using a pre-trained model.

    Args:
        model (object): The pre-trained model used for caption generation.
        img_path (str): The path to the image file.
        tokenizer (object): The tokenizer used for tokenizing the caption.
        max_length (int): The maximum length of the generated caption.

    Returns:
        str: The generated caption for the image.
    """
    # find the image using os
    image_path = os.path.abspath(img_path)
    
    # extract features
    extracted_features = extract_features(image_path)
    y_pred = pred_caption(model, extracted_features, tokenizer, max_length)
    # Make a fig, axs that holds the image and the caption
    
    # Make a figure with subplots to hold the image and caption
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.imshow(Image.open(image_path))
    axs.axis('off')
    axs.set_title(y_pred[9:-7], fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{img_path}_Prediction.png')
    plt.show()
    return y_pred[9:-7]

# Now let's calculate the BLEU Score for the model
# Use the test set to evaluate the model
def evaluate_model(model, captions, features, tokenizer, max_length, num):
    """
    Evaluate the image captioning model by generating captions for images and calculating BLEU scores.

    Args:
        model (object): The trained image captioning model.
        captions (dict): A dictionary containing image IDs as keys and corresponding captions as values.
        features (dict): A dictionary containing image IDs as keys and corresponding image features as values.
        tokenizer (object): The tokenizer used to tokenize the captions.
        max_length (int): The maximum length of the generated captions.
        num (int): The number of images to evaluate.

    Returns:
        None
    """
    actual, predicted = list(), list()
    for key, caps in captions.items():
        y_pred = pred_caption(model, features[key], tokenizer, max_length)
        references = [d.split() for d in caps]
        actual.append(references)
        predicted.append(y_pred.split())
        for i in range(len(caps)):
            print('Image ID: %s\nActual caption: %s\nPredicted caption: %s\n' % (key, caps[i], y_pred))
        # Make a figure with subplots to hold the image and caption
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'Flicker8k', 'Images', key + '.jpg')
        axs.imshow(Image.open(img_path))
        axs.axis('off')
        axs.set_title(y_pred[9:-7], fontsize=12, ha='center')
        plt.tight_layout()
        plt.savefig(f'{key}_Prediction.png')
        plt.show()
    # BLEU Score Evaluation
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
        

test = img_ids[split:]
#lets take only 10 images for testing


def num_of_imgs(num):
    """
    Selects a specified number of random images from the test set and evaluates the model on them.

    Args:
        num (int): The number of random images to select from the test set.

    Returns:
        None
    """
    # Gather random images form the test set
    random_test = random.sample(test, num)
    #print(test)
    test_captions = {}
    for key in random_test:
        test_captions[key] = map[key]
    evaluate_model(model, test_captions, image_features, tokenizer, max_length, num)
    
    
    

# Main function
if __name__ == '__main__':
    # Two cases: One to test using num_of_imgs, second case to pass in their own image path
    # Case statements:
    
    response = input("Hi Welcome to our image caption generator model. Would you like to test our model using our test set? (y/n): ")
    if response == 'y':
        print()
        num = input("How many images would you like to test? (1-10): ")
        num_of_imgs(int(num))
    else:
        print()
        img_path = input("Please enter the image path: ")
        print(pred_caption_any_img(model, img_path, tokenizer, max_length))

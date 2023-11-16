import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout, add
from tensorflow.keras.layers import LayerNormalization
import pickle
import os



Model.load('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
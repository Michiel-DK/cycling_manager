import pandas as pd
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Concatenate, Add, Masking, GRU, RepeatVector, Dot, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MAE, MSE, RootMeanSquaredError, Recall, Precision, Accuracy, CategoricalAccuracy
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# define the encoder and decoder -> clean
def encoder(encoder_features):
    y = Masking(mask_value = -1000.)(encoder_features)
    #each LSTM unit returning a sequence of 6 outputs, one for each time step in the input data
    y = LSTM(units=28, return_sequences=True, activation='tanh')(y)
    y = LSTM(units=14, return_sequences=True, activation='tanh')(y)
    #output one time step from the sequence for each time step in the input but process 5 outputs of the input sequence at a time
    #y = TimeDistributed(Dense(units=5, activation='tanh'))(y)
    y = LSTM(units=14, return_sequences=False, activation='tanh')(y)
    y = RepeatVector(21)(y)
    return y

def decoder(decoder_features, encoder_outputs):
    x = Concatenate(axis=-1)([decoder_features, encoder_outputs])
    # x = Add()([decoder_features, encoder_outputs]) 
    x = Masking(mask_value = 0)(x)
    x = TimeDistributed(Dense(units=32, activation='relu'))(x)
    x = TimeDistributed(Dense(units=16, activation='relu'))(x)
    x = TimeDistributed(Dense(units=6, activation='relu'))(x)
    y = TimeDistributed(Dense(units=2, activation='softmax'))(x)
    return y
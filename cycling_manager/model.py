import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from colorama import Fore, Style

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Concatenate, Add, Masking, GRU, RepeatVector, Dot, Bidirectional, LayerNormalization, Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MAE, MSE, RootMeanSquaredError, Recall, Precision, Accuracy, CategoricalAccuracy
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# define the encoder and decoder -> clean
def encoder(encoder_features):    
    y = Masking(mask_value = -1000.)(encoder_features)
    #each LSTM unit returning a sequence of 6 outputs, one for each time step in the input data
    y = LSTM(units=15, dropout=0.2, return_sequences=True, activation='tanh')(y)
    y = LayerNormalization()(y)
    y = LSTM(units=30, dropout=0.2, return_sequences=True, activation='tanh')(y)
    y = LayerNormalization()(y)
    #output one time step from the sequence for each time step in the input but process 5 outputs of the input sequence at a time
    #y = TimeDistributed(Dense(units=5, activation='tanh'))(y)
    y = LSTM(units=14, dropout=0.2, return_sequences=False, activation='tanh')(y)
    y = RepeatVector(21)(y)
    return y

def decoder(decoder_features, encoder_outputs):
    x = Concatenate(axis=-1)([decoder_features, encoder_outputs])
    # x = Add()([decoder_features, encoder_outputs]) 
    x = Masking(mask_value = -1000.)(x)
    x = TimeDistributed(Dense(units=32, activation='relu'))(x)
    x = TimeDistributed(Dense(units=16, activation='relu'))(x)
    x = TimeDistributed(Dense(units=6, activation='relu'))(x)
    y = TimeDistributed(Dense(units=1, activation='linear'))(x)
    return y

def combine_model(X_encoder, X_decoder):
    
    encoder_features = Input(shape=X_encoder.shape[1:])
    decoder_features = Input(shape=X_decoder.shape[1:])
    encoder_outputs = encoder(encoder_features)
    decoder_outputs = decoder(decoder_features, encoder_outputs)
    model = Model([encoder_features, decoder_features], decoder_outputs)
    
    print(Fore.YELLOW + f"\nCombined model..." + Style.RESET_ALL)
    
    return model

def compile_model(model):
    
    initial_learning_rate = 0.01

    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.5)

    adam = Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics='mean_absolute_error')
    
    print(Fore.YELLOW + f"\Compile model..." + Style.RESET_ALL)

    return model

def train_model(model: Model,
                x_encoder: np.ndarray,
                x_decoder: np.ndarray,
                y: np.ndarray,
                batch_size=128,
                patience=20,
                validation_split=0.3,
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit([x_encoder, x_decoder],
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=1000,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1)


    return model, history

def evaluate_model(model: Model,
                   X_encoder: np.ndarray,
                   X_decoder: np.ndarray,
                   y_decoder: np.ndarray,
                   batch_size=128) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model {len(X_encoder.shape)}" + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=[X_encoder, X_decoder],
        y=y_decoder,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)
    
    loss = metrics["loss"]
    mae = metrics["mean_absolute_error"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

    return metrics
import pandas as pd
import numpy as np

from colorama import Fore, Style

from cycling_manager.sequences import *
from cycling_manager.preprocess import *
from cycling_manager.model import *

def preprocess(
    start: int,
    end: int
) -> pd.DataFrame:
    
    """ full preprocess data + split """
    
    df = preprocess(get_data())
    
    train, test = split(df, start=start, end=end)
    
    print(Fore.BLUE + f"\nProcessing done..." + Style.RESET_ALL)
    
    return df, train, test

def get_seq(train, test, df, maxlen):
    
    X_enc_ss, X_dec_ss = get_scaler(80, df, train)
    
    X_encoder_train, X_decoder_train, y_decoder_train = get_sequences(maxlen, df, train, X_enc_ss, X_dec_ss)
    X_encoder_test, X_decoder_test, y_decoder_test = get_sequences(maxlen, df, test, X_enc_ss, X_dec_ss)
    
    print(Fore.GREEN + f"\nSequencing done..." + Style.RESET_ALL)
    
    return X_encoder_train, X_decoder_train, y_decoder_train, X_encoder_test, X_decoder_test, y_decoder_test 

def train(start=2017, end=2022, maxlen=80):
    
    #get train test data
    df, train, test = preprocess(start, end)
    
    #get train sequences
    X_encoder_train, X_decoder_train, y_decoder_train = get_seq(train, test, df, maxlen)
    
    #create model
    model = combine_model(X_decoder_train, X_decoder_train)

    #compile model
    model = model.compile(model)
    
    model, history = train_model(model, X_encoder_train, X_decoder_train, y_decoder_train\
        batch_size=128, patience=10, validation_split=0.2, validation_data=None)
    
    
    
    
    
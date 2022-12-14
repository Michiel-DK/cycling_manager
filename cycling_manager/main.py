import pandas as pd
import numpy as np

from colorama import Fore, Style

from cycling_manager.sequences import *
from cycling_manager.preprocess import *
from cycling_manager.model import *
from cycling_manager.registry import *

def preproc(
        start : int,
        end : int
    ) -> pd.DataFrame:
    
    """ full preprocess data + split """
    
    df = preprocess(get_data())
    
    train, test = split(df, start=start, end=end)
    
    print(Fore.BLUE + f"\nProcessing done..." + Style.RESET_ALL)
    
    return df, train, test

def get_seq(
        values : pd.DataFrame, 
        df : pd.DataFrame, 
        maxlen : int) -> Tuple:
    
    """get scaler based on given df and maxlength of sequences"""
    
    X_enc_ss, X_dec_ss = get_scaler(maxlen, df, values)
    
    print(Fore.GREEN + f"\Scaler done..." + Style.RESET_ALL)
    
    X_encoder, X_decoder, y_decoder = get_sequences(maxlen, df, values, X_enc_ss, X_dec_ss)
    
    print(Fore.GREEN + f"\nSequencing done..." + Style.RESET_ALL)
    
    return X_encoder, X_decoder, y_decoder

def train(start:int = 2017,
          end: int = 2022, 
          maxlen: int = 80) -> None:
    
    """train model and save locally"""
    
    #get train test data
    df, train, test = preproc(start, end)
    
    #get train sequences
    X_encoder_train, X_decoder_train, y_decoder_train = get_seq(train, df, maxlen)
    
    print(Fore.CYAN + f"\n{X_encoder_train.shape, X_decoder_train.shape, y_decoder_train.shape}" + Style.RESET_ALL)
    
    #create model
    model = combine_model(X_encoder_train, X_decoder_train)

    #compile model
    model = compile_model(model)
    
    model, history = train_model(model, X_encoder_train, X_decoder_train, y_decoder_train,\
        batch_size=128, patience=5, validation_split=0.2, validation_data=None)
    
    metrics = np.min(history.history)
    
    params = dict(
        # Model parameters
        start=start,
        end=end,
        maxlen=maxlen)
    
    metrics = dict(
        accuracy='precision'
    )
    
    save_model(model, params=params, metrics=metrics)
    
    return None
    

if __name__=='__main__':
    train(start=2017, end=2022, maxlen=80)
    
    
    
    
    
    
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
    ) -> Tuple:
    
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

def train(df:pd.DataFrame,
          train:pd.DataFrame,
          start:int = 2017,
          end: int = 2022, 
          maxlen: int = 80) -> None:
    
    """train model and save locally"""
    
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
        mse=metrics
    )
    
    save_model(model, params=params, metrics=metrics)
    
    return None

def evaluate(
    df : pd.DataFrame,
    test : pd.DataFrame,
    maxlen: int
):
    
    model = load_model()
    
    X_encoder_test, X_decoder_test, y_decoder_test = get_seq(test, df, maxlen)

    metrics_dict = evaluate_model(model, X_encoder_test, X_decoder_test, y_decoder_test)
    mae = metrics_dict["mean_absolute_error"]

    # Save evaluation
    params = dict(
        model_version=get_model_version(),

        # Package behavior
        context="evaluate",

        # Data source

        row_count=len(X_encoder_test)
    )

    save_model(params=params, metrics=dict(mae=mae))

    return mae
    

if __name__=='__main__':
        #get train test data
        try:
            start = 2010
            end = 2021
            df, train_df, test_df = preproc(start, end)
            #train(df, train_df, start, end, maxlen=80)
            mae = evaluate(df, test_df, 80)
            print(mae)
        except:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
        
    
    
    
    
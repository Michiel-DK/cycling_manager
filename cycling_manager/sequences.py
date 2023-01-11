import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from keras_preprocessing.sequence import pad_sequences

from colorama import Fore, Style


def get_sequence(df : pd.DataFrame, name, year, tour, maxlen=40, img=False, binary=False):
    
    #get tour data
    if year != 2000:
        tour_data = df[(df['name'] == name) & (df['year'] == year) & (df['race_name'] == tour)].sort_values(by='date')
        if binary == True:
            y_decoder = tour_data[['date', 'result_bin']].set_index('date')
        else:
            y_decoder = tour_data[['date', 'result']].set_index('date')
        X_decoder = tour_data[['date', 'distance', 'ProfileScore:', 'Vert. meters:', 'Startlist quality score:', 'parcours_type_num']].set_index('date')
        
        season_data = df[(df['name'] == name) & (df['date'] < min(tour_data['date'])) & (df['date'] >= min(tour_data['date']) - datetime.timedelta(weeks=102))].sort_values(by='date')
        season_data = season_data[season_data['result']<=60]
        X_encoder = season_data[['date', 'adjusted_points','result','distance', 'ProfileScore:', 'Vert. meters:', 'Startlist quality score:', 'parcours_type_num', 'icon_bin','Avg. speed winner:', 'types_bin', 'gt_binary']].set_index('date')
        performance = X_encoder.pivot_table('adjusted_points', 'date','parcours_type_num').shift(1).fillna(0).cumsum().reset_index()
        X_encoder = performance.merge(X_encoder, on='date').set_index('date').drop(columns='adjusted_points').rename(columns={1.0 : 'fl', 2.0:'hi_fl', 3.0: 'hi_hi', 4.0:'mo_fl', 5.0:'mo_mo'})
        
        for i in ['fl', 'hi_fl', 'hi_hi', 'mo_fl', 'mo_mo']:
            if i not in X_encoder.columns:
                X_encoder[i] = 0.0
                
        #X_encoder = X_encoder[['fl', 'mo_fl', 'hi_fl', 'hi_hi', 'mo_mo', 'distance','result', 'ProfileScore:','Vert. meters:', 'Startlist quality score:', 'parcours_type_num', 'icon_bin','Avg. speed winner:', 'types_bin', 'gt_binary']]
        X_encoder = X_encoder[['distance','result','Vert. meters:', 'Startlist quality score:','icon_bin','Avg. speed winner:', 'gt_binary']]
        if X_encoder.isna().sum().sum() > 0:
            X_encoder.dropna(inplace=True)
            print(Fore.RED + f"\n dropped nan for {name, year, tour}" + Style.RESET_ALL)
    
        else:
            pass
        
    
    else:
        pass
    if img:
        return y_decoder, tour_data.race_ref.unique(), season_data[['race_ref', 'result']].tail(maxlen), season_data.tail(maxlen).race_ref.unique()
    return X_encoder.tail(maxlen), X_decoder, y_decoder

def get_scaler(maxlen, df, riders):
    
    X_encoder_ls = []
    X_decoder_ls = []
    y_ls = []


    for rider, year, tour in riders.values:
        
        if year != 2000:
            
            #X_encoder, X_decoder, y_decoder, y_encoder = get_sequence(df, rider, year, tour)
            X_encoder, X_decoder, y = get_sequence(df, rider, year, tour, maxlen)
            
            X_encoder_ls.append(X_encoder)
            X_decoder_ls.append(X_decoder)
            y_ls.append(y)
            
    X_encoder_scdf = pd.concat(X_encoder_ls)
    X_decoder_scdf = pd.concat(X_decoder_ls)
    y_train_scdf = pd.concat(y_ls)
    
    X_enc_mm = MinMaxScaler()
    X_enc_mm.fit(X_encoder_scdf)
    
    X_dec_mm = MinMaxScaler()
    X_dec_mm.fit(X_decoder_scdf)
    
    #y_train_mm = MinMaxScaler()
    #y_train_mm.fit(y_train_scdf)

            
    return X_enc_mm, X_dec_mm#, y_train_mm

def get_sequences(maxlen, df, riders, enc_scaler, dec_scaler, binary):
    
    X_encoder_ls = []
    X_decoder_ls = []
    #y_encoder_ls = []
    y_decoder_ls = []
    
    i=0


    for rider, year, tour in riders.values:
        
        if year != 2000:
            
        
            
            #try:
            
                X_encoder, X_decoder, y_decoder = get_sequence(df, rider, year, tour, maxlen, binary=binary)
                print(X_encoder.shape)
                #X_encoder, X_decoder, y_decoder, y_encoder = get_sequence(df, rider, year, tour)
                
                if X_encoder.shape == (0,7):
                    print(Fore.RED + f"\n X_encoder empty for {rider, year, tour}" + Style.RESET_ALL)
                
                else:
                    X_encoder = enc_scaler.transform(X_encoder)
                    X_decoder = dec_scaler.transform(X_decoder)
                    #y = y_train_mm.transform(y_decoder)
                    
                    X_encoder_pad = pad_sequences(X_encoder.T, maxlen=maxlen, dtype='float', padding='pre', value=-1000.).T
                    #y_encoder_pad = pad_sequences(y_encoder.to_numpy().T, maxlen=maxlen, dtype='float', padding='pre', value=-1000.).T
                    
                    X_decoder_pad = pad_sequences(X_decoder.T, maxlen=21, dtype='float', padding='post', value=-1000.).T
                    y_decoder_pad = pad_sequences(y_decoder.to_numpy().T, maxlen=21, dtype='float', padding='post', value=-1000.).T
                    #X_decoder_pad = pad_sequences(X_decoder_pad.T, maxlen=maxlen, dtype='float', padding='pre', value=-1000.).T
                    
                    #y_decoder_pad = pad_sequences(y_decoder.T, maxlen=21, dtype='float', padding='post', value=-1000.).T
                    #y_pad = pad_sequences(y_decoder_pad.T, maxlen=maxlen, dtype='float', padding='pre', value=-1000.).T
                        
                    X_encoder_ls.append(X_encoder_pad)
                    X_decoder_ls.append(X_decoder_pad)
                    #y_encoder_ls.append(y_encoder_pad)
                    y_decoder_ls.append(y_decoder_pad)
                    #y_ls.append(y_pad)
                    
                    #print(X_encoder_pad.shape, X_decoder_pad.shape, y_pad.shape)
                
                #except ValueError:
                    #i+=1
                    #print(f"ValueError for {i, rider, year, tour}")
    
    return np.array(X_encoder_ls), np.array(X_decoder_ls), np.array(y_decoder_ls)
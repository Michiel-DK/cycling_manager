import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import tensorflow as tf
import cv2

from keras_preprocessing.sequence import pad_sequences

from colorama import Fore, Style


def get_full_sequence(df : pd.DataFrame, name, year, tour, maxlen_num=40, maxlen_img=20, img=False, binary=False, y_encoder = True, num_cap=9999, img_cap=20):
    
    """Function that returns full sequence for numerical and imgage model"""
    
    #if img return full sequence
    tour_data = df[(df['name'] == name) & (df['year'] == year) & (df['race_name'] == tour)].sort_values(by='date')
        
        # return correct target for y_decoder sequence
    if binary == True:
        y_decoder = tour_data[['date', 'result_bin']].set_index('date')
    else:
        y_decoder = tour_data[['date', 'result']].set_index('date')
        
    # return values for X_decoder 
    X_decoder_num = tour_data[['date', 'distance', 'ProfileScore:', 'Vert. meters:', 'Startlist quality score:', 'parcours_type_num']].set_index('date')
        
    season_data = df[(df['name'] == name) & (df['date'] < min(tour_data['date'])) & (df['date'] >= min(tour_data['date']) - datetime.timedelta(weeks=102))].sort_values(by='date')
        
    #getting X_encoder
    X_encoder = season_data[['date', 'adjusted_points','result','distance', 'ProfileScore:', 'Vert. meters:', 'Startlist quality score:', 'parcours_type_num', 'icon_bin','Avg. speed winner:', 'types_bin', 'gt_binary']].set_index('date')
    performance = X_encoder.pivot_table('adjusted_points', 'date','parcours_type_num').shift(1).fillna(0).cumsum().reset_index()
    X_encoder = performance.merge(X_encoder, on='date').set_index('date').drop(columns='adjusted_points').rename(columns={1.0 : 'fl', 2.0:'hi_fl', 3.0: 'hi_hi', 4.0:'mo_fl', 5.0:'mo_mo'})
            
    for i in ['fl', 'hi_fl', 'hi_hi', 'mo_fl', 'mo_mo']:
        if i not in X_encoder.columns:
            X_encoder[i] = 0.0
        
    X_encoder = X_encoder[['fl', 'mo_fl', 'hi_fl', 'hi_hi', 'mo_mo', 'distance','result', 'ProfileScore:','Vert. meters:', 'Startlist quality score:', 'parcours_type_num', 'icon_bin','Avg. speed winner:', 'types_bin', 'gt_binary']]
        
        #split for typing
    X_encoder_num = X_encoder[X_encoder['result']<=num_cap].tail(maxlen_num)
    X_encoder_num_no_y = X_encoder_num.drop(columns='result')
    y_encoder_num = X_encoder_num.result
        
    if X_encoder.isna().sum().sum() > 0:
        print(Fore.RED + f"\n passed for {name, year, tour} due to nan" + Style.RESET_ALL)
        pass
    
    #if no image needed
    if img == False:
        
        if y_encoder:
            return X_decoder_num, y_decoder, X_encoder_num_no_y, y_encoder_num
            
        else:
            return X_decoder_num, y_decoder,X_encoder_num
    
    #if image needed
    else: 
        if y_encoder:
            
            X_decoder_img_ls = tour_data.race_ref
            
            X_encoder_img = X_encoder[X_encoder['result']<=img_cap].tail(maxlen_img)
            
            X_encoder_img_ls = X_encoder_img.race_ref
            
            y_encoder_img_ls = X_encoder_img.result
            
            
            return X_decoder_num, y_decoder, X_encoder_num_no_y, y_encoder_num, X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls
            
        else:
            return X_decoder_num, y_decoder, X_encoder_num, X_decoder_img_ls, X_encoder_img_ls
        

def get_images(X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls):
    
    """Function to fetch images and set in sequences"""

    season_ls_img = []
    season_y_img = []
    to_drop_ls = []
    base_path = '../raw_data/img_300/'
    
    for season, result in zip(X_encoder_img_ls, y_encoder_img_ls):
        
        season = [base_path+"_".join(race.split('/')[1:])+'.jpg' for race in season]
    
        img_ls = []
        result_ls = []
    
        for race in season:
            img = cv2.imread(race)
            try:
                img = tf.convert_to_tensor(img)
                img_ls.append(img)
                result_ls.append(result)
            except:
                to_drop_ls.append('race/'+race.split('/')[-1].split('.')[0].replace('_', '/'))
                pass
        
            season_ls_img.append(np.array(img_ls))
            season_y_img.append(np.array(result_ls))
    
    season_ls_img = np.array(season_ls_img)
    season_y_img = np.array(season_y_img)
    to_drop_ls = list(dict.fromkeys(to_drop_ls))
    
    for season in X_decoder_img_ls:
        season = [base_path+"_".join(race.split('/')[1:])+'.jpg' for race in season]
    
        img_ls = []
        
        for race in season:
            img = cv2.imread(race)
            try:
                img = tf.convert_to_tensor(img)
                img_ls.append(img)
            except:
                print(race)
            
        tour_ls_img.append(np.array(img_ls))
        
    tour_ls_img = np.array(tour_ls_img)
    
    return season_ls_img, tour_ls_img, season_y_img, to_drop_ls
    
            

        

def get_sequence(df : pd.DataFrame, name, year, tour, maxlen=40, img=False, binary=False):
    
    #get tour data
        tour_data = df[(df['name'] == name) & (df['year'] == year) & (df['race_name'] == tour)].sort_values(by='date')
        if tour_data.shape[0] != 0:
            if binary == True:
                y_decoder = tour_data[['date', 'result_bin']].set_index('date')
            else:
                y_decoder = tour_data[['date', 'result']].set_index('date')
            X_decoder = tour_data[['date', 'distance', 'ProfileScore:', 'Vert. meters:', 'Startlist quality score:', 'parcours_type_num']].set_index('date')
            
            season_data = df[(df['name'] == name) & (df['date'] < min(tour_data['date'])) & (df['date'] >= min(tour_data['date']) - datetime.timedelta(weeks=102))].sort_values(by='date')
            
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
            
            
            if img:
                return y_decoder, tour_data.race_ref.unique(), season_data[['race_ref', 'result']].tail(maxlen), season_data.tail(maxlen).race_ref.unique()
            return X_encoder.tail(maxlen), X_decoder, y_decoder
        else:
            return 

def get_scaler(maxlen, df, riders):
    
    X_encoder_ls = []
    X_decoder_ls = []
    y_ls = []


    for rider, year, tour in riders.values:
        
        if year != 2000:
            
            #X_encoder, X_decoder, y_decoder, y_encoder = get_sequence(df, rider, year, tour)
            try:
                X_encoder, X_decoder, y = get_sequence(df, rider, year, tour, maxlen)
                
                X_encoder_ls.append(X_encoder)
                X_decoder_ls.append(X_decoder)
                y_ls.append(y)
            except TypeError as t:
                print(f'{t} -- {rider, year, tour}')
            
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
                try:
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
                except TypeError as t:
                    print(f'{t} -- {rider, year, tour}')
                    #y_ls.append(y_pad)
                    
                    #print(X_encoder_pad.shape, X_decoder_pad.shape, y_pad.shape)
                
                #except ValueError:
                    #i+=1
                    #print(f"ValueError for {i, rider, year, tour}")
    
    return np.array(X_encoder_ls), np.array(X_decoder_ls), np.array(y_decoder_ls)
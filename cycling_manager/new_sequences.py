import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import tensorflow as tf
import cv2

from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import resnet50


from colorama import Fore, Style


def get_num_sequence(df : pd.DataFrame, name, year, tour, maxlen_num=40, maxlen_img=20, img=False, binary=False, y_encoder = True, num_cap=9999, img_cap=20):
    
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
    X_encoder_merged = performance.merge(X_encoder, on='date').set_index('date').drop(columns='adjusted_points').rename(columns={1.0 : 'fl', 2.0:'hi_fl', 3.0: 'hi_hi', 4.0:'mo_fl', 5.0:'mo_mo'})
            
    for i in ['fl', 'hi_fl', 'hi_hi', 'mo_fl', 'mo_mo']:
        if i not in X_encoder_merged.columns:
            X_encoder_merged[i] = 0.0
        
    X_encoder = X_encoder_merged[['fl', 'mo_fl', 'hi_fl', 'hi_hi', 'mo_mo', 'distance','result', 'ProfileScore:','Vert. meters:', 'Startlist quality score:', 'parcours_type_num', 'icon_bin','Avg. speed winner:', 'types_bin', 'gt_binary']]
    X_encoder.dropna(inplace=True)
        #split for typing
    X_encoder_num = X_encoder[X_encoder['result']<=num_cap].tail(maxlen_num)
    X_encoder_num_no_y = X_encoder_num.drop(columns='result')
    y_encoder_num = X_encoder_num[['result']]
        
    if X_encoder.isna().sum().sum() > 0:
        print(Fore.RED + f"\n passed for {name, year, tour} due to nan" + Style.RESET_ALL)
        pass
    else:
        X_decoder_num = tf.convert_to_tensor(pad_sequences(X_decoder_num.to_numpy().T, maxlen=21, dtype='float', padding='post', value=-1000.).T)
        y_decoder = tf.convert_to_tensor(pad_sequences(y_decoder.to_numpy().T, maxlen=21, dtype='float', padding='post', value=-1000.).T)
        X_encoder_num = tf.convert_to_tensor(pad_sequences(X_encoder_num.to_numpy().T, maxlen=maxlen_num, dtype='float', padding='post', value=-1000.).T)
        X_encoder_num_no_y = tf.convert_to_tensor(pad_sequences(X_encoder_num_no_y.to_numpy().T, maxlen=maxlen_num, dtype='float', padding='post', value=-1000.).T)
        
        y_encoder_num = tf.convert_to_tensor(pad_sequences(y_encoder_num.to_numpy().T, maxlen=maxlen_num, dtype='float', padding='post', value=-1000.).T)
        
        #if no image needed
        if img == False:
            
            if y_encoder:
                return X_decoder_num, y_decoder, X_encoder_num_no_y, y_encoder_num
                
            else:
                return X_decoder_num, y_decoder,X_encoder_num
        
        #if image needed
        else:
            X_decoder_img_ls = tour_data.race_ref
                
            X_encoder_img = X_encoder[X_encoder['result']<=img_cap].tail(maxlen_img)
            X_encoder_img_ls = list(pd.merge(X_encoder_img, season_data, on='date', how='inner').race_ref)
                
            y_encoder_img_ls = list(X_encoder_img.result)
            
            if y_encoder == True:
                
                return X_decoder_num, y_decoder, X_encoder_num_no_y, y_encoder_num, X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls
                
            else:
                return X_decoder_num, y_decoder, X_encoder_num, X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls
        

def get_images(X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls):
    
    """Function to fetch images and set in sequences"""

    season_ls_img = []
    season_y_img = []
    to_drop_ls = []
    tour_ls_img = []
    base_path = '../raw_data/img_300/'
    
    X_encoder_img_ls = [base_path+"_".join(race.split('/')[1:])+'.jpg' for race in X_encoder_img_ls]
    X_decoder_img_ls = [base_path+"_".join(race.split('/')[1:])+'.jpg' for race in X_decoder_img_ls]    

    img_ls = []
    result_ls = []
    
    for race ,result in zip(list(X_encoder_img_ls), list(y_encoder_img_ls)):
        img = cv2.imread(race)
        try:
            img = tf.convert_to_tensor(img, dtype=tf.int16)
            result = tf.convert_to_tensor(result, dtype=tf.float16)
            img_ls.append(img)
            result_ls.append(result)
        except:
            to_drop_ls.append('race/'+race.split('/')[-1].split('.')[0].replace('_', '/'))
            pass
        
    # season_ls_img.append(img_ls)
    # season_y_img.append(result_ls)
    # import ipdb; ipdb.set_trace()
    # X_encoder_img = tf.ragged.stack(season_ls_img).to_tensor()
    # y_encoder_img = tf.ragged.stack(season_y_img).to_tensor()
    X_encoder_img = img_ls
    y_encoder_img = result_ls
    to_drop_ls = list(dict.fromkeys(to_drop_ls))
    
    img_ls = []
        
    for race in X_decoder_img_ls:
        img = cv2.imread(race)
        try:
            img = tf.convert_to_tensor(img)
            img_ls.append(img)
        except:
            print(race)
            
    tour_ls_img.append(np.array(img_ls))
        
    #X_decoder_img = tf.ragged.stack(tour_ls_img).to_tensor()
    X_decoder_img = img_ls
            
    return X_encoder_img, X_decoder_img, y_encoder_img, to_drop_ls



def get_full_sequence(df, riders, maxlen_num=40, maxlen_img=20, img=True, binary=True, y_encoder = True, num_cap=9999, img_cap=20, resnet=False):
    
    X_decoder_num_ls = []
    y_decoder_ls = []
    X_encoder_num_no_y_ls = []
    y_encoder_num_ls = []
    
    X_img_decoder_ls = []
    X_img_encoder_ls = []
    y_img_encoder_ls = []
    
    for rider, year, tour in riders.values:
        X_decoder_num, y_decoder, X_encoder_num_no_y, y_encoder_num, X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls=\
                get_num_sequence(df, rider, year, tour, maxlen_num = maxlen_num, maxlen_img=maxlen_img, img=img, binary=binary, y_encoder=y_encoder, num_cap=num_cap, img_cap=img_cap)
                
        if len(X_encoder_img_ls) == 0:
            pass
        
        else:
                                 
            X_encoder_img, X_decoder_img, y_encoder_img, to_drop_ls = get_images(X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls)
            
            X_decoder_num_ls.append(X_decoder_num)
            y_decoder_ls.append(y_decoder)
            
            X_encoder_num_no_y_ls.append(X_encoder_num_no_y)
            y_encoder_num_ls.append(y_encoder_num)
            
            X_img_decoder_ls.append(X_decoder_img)
            X_img_encoder_ls.append(X_encoder_img)
            y_img_encoder_ls.append(y_encoder_img)
    
    X_decoder_num_ls = tf.convert_to_tensor(X_decoder_num_ls)
    y_decoder_ls = tf.convert_to_tensor(y_decoder_ls)
    X_encoder_num_no_y_ls = tf.convert_to_tensor(X_encoder_num_no_y_ls)
    y_encoder_num_ls = tf.convert_to_tensor(y_encoder_num_ls)
    
    X_img_decoder_ls = tf.ragged.stack(X_img_decoder_ls).to_tensor()
    X_img_encoder_ls = tf.ragged.stack(X_img_encoder_ls).to_tensor()
    y_img_encoder_ls = tf.ragged.stack(y_img_encoder_ls).to_tensor()
    
    if resnet:
        X_img_decoder_ls = resnet50.preprocess_input(X_img_decoder_ls)
        X_img_encoder_ls = resnet50.preprocess_input(X_img_encoder_ls)
        
    
    print(X_decoder_num_ls.shape, y_decoder_ls.shape, X_encoder_num_no_y_ls.shape, y_encoder_num_ls.shape, X_img_decoder_ls.shape, X_img_encoder_ls.shape, y_img_encoder_ls.shape)
        
    return X_decoder_num_ls, y_decoder_ls, X_encoder_num_no_y_ls, y_encoder_num_ls, X_img_decoder_ls, X_img_encoder_ls, y_img_encoder_ls
                                 
    
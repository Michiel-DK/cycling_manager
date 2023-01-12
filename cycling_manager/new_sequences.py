import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import tensorflow as tf
import cv2

from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import resnet50


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
    X_encoder_merged = performance.merge(X_encoder, on='date').set_index('date').drop(columns='adjusted_points').rename(columns={1.0 : 'fl', 2.0:'hi_fl', 3.0: 'hi_hi', 4.0:'mo_fl', 5.0:'mo_mo'})
            
    for i in ['fl', 'hi_fl', 'hi_hi', 'mo_fl', 'mo_mo']:
        if i not in X_encoder.columns:
            X_encoder[i] = 0.0
        
    X_encoder = X_encoder_merged[['fl', 'mo_fl', 'hi_fl', 'hi_hi', 'mo_mo', 'distance','result', 'ProfileScore:','Vert. meters:', 'Startlist quality score:', 'parcours_type_num', 'icon_bin','Avg. speed winner:', 'types_bin', 'gt_binary']]
        
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
            X_encoder_img_ls = list(pd.merge(X_encoder_img, season_data, on='date', how='inner').race_ref)
            
            y_encoder_img_ls = list(X_encoder_img.result)
            
            
            return X_decoder_num, y_decoder, X_encoder_num_no_y, y_encoder_num, X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls
            
        else:
            return X_decoder_num, y_decoder, X_encoder_num, X_decoder_img_ls, X_encoder_img_ls
        

def get_images(X_decoder_img_ls, X_encoder_img_ls, y_encoder_img_ls, resnet=False):
    
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
            print(race)
            img = cv2.imread(race)
            try:
                img = tf.convert_to_tensor(img, dtype=tf.int16)
                result = tf.convert_to_tensor(result, dtype=tf.float16)
                img_ls.append(img)
                result_ls.append(result)
            except:
                to_drop_ls.append('race/'+race.split('/')[-1].split('.')[0].replace('_', '/'))
                pass
        
    season_ls_img.append(np.array(img_ls))
    season_y_img.append(np.array(result_ls))
    
    X_encoder_img = tf.ragged.stack(season_ls_img).to_tensor()
    y_encoder_img = tf.ragged.stack(season_y_img).to_tensor()
    
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
        
    X_decoder_img = tf.ragged.stack(tour_ls_img).to_tensor()
    
    if resnet:
        X_encoder_img = resnet50.preprocess_input(X_encoder_img)
        X_decoder_img = resnet50.preprocess_input(X_decoder_img)
    
    return X_encoder_img, X_decoder_img, y_encoder_img, to_drop_ls


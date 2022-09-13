from importlib import import_module
import pandas as pd
import numpy as np
import datetime
import colored
from colored import stylize
import os

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_data(local=True) -> pd.DataFrame:
    if local:
        merged1 = pd.read_csv(f'{os.getenv("LOCAL_PATH")}/raw_data/merged_clean.csv', index_col=0)
        merged2 = pd.read_csv(f'{os.getenv("LOCAL_PATH")}/raw_data/merged_clean_2.csv', index_col=0)
        merged3 = pd.read_csv(f'{os.getenv("LOCAL_PATH")}/raw_data/merged_clean_tdf2022.csv', index_col=0)
        merged4 = pd.read_csv(f'{os.getenv("LOCAL_PATH")}/raw_data/merged_clean_10s.csv', index_col=0)
        merged5 = pd.read_csv(f'{os.getenv("LOCAL_PATH")}/raw_data/merged_clean_vuelta2022.csv', index_col=0)
        
        merged = pd.concat([merged1, merged2, merged3, merged4, merged5], ignore_index=True)
        
        merged.drop_duplicates(inplace=True)

        merged.dropna(subset=['ProfileScore:', 'Vert. meters:', 'Distance:'], how='all', inplace=True)
        merged.reset_index(inplace=True, drop=True)
        
        merged['date'] = pd.to_datetime(merged['date'])
        
        print(stylize(f"full df shape {merged.shape}", colored.fg("green")))
        
        return merged
    

def preprocess(df:pd.DataFrame) -> pd.DataFrame:
    
    def vert_meters(df:pd.DataFrame) -> pd.DataFrame:
    #get data
        merged = df.copy()
        
        #preprocess vertical meters
        no_na = merged.dropna(subset=['ProfileScore:', 'Vert. meters:', 'distance'], how='any')
        
        X  = no_na[['Distance:', 'ProfileScore:']]
        y = no_na['Vert. meters:']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()

        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        tree = DecisionTreeRegressor()

        tree.fit(X_train_scaled, y_train)

        print(stylize(f"Score vertical meters - model: {tree} score: {tree.score(X_test_scaled, y_test)}", colored.fg("yellow")))
        
        # add to df
        vert_na = merged[merged['Vert. meters:'].isna()]
        vert_na.dropna(subset=['Distance:', 'ProfileScore:'], how='any', inplace=True)

        X_pred = vert_na[['Distance:', 'ProfileScore:']]

        X_pred_scaled = scaler.transform(X_pred)

        vert_na['predicted_vert'] = tree.predict(X_pred_scaled)
        
        return vert_na
    
    def profile_score(df:pd.DataFrame) -> pd.DataFrame:
        
        merged = df.copy()
        
        #drop where either profile score or vert meters are missing
        no_na = merged.dropna(subset=['ProfileScore:', 'Vert. meters:'], how='any')
        print(no_na.shape)

        #drop where score below 10 -> arbitrary point
        no_na = no_na[no_na['ProfileScore:'] > 10]
        
        # profile score via vert and distance
        X  = no_na[['Distance:', 'Vert. meters:']]
        y = no_na['ProfileScore:']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()

        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        tree = DecisionTreeRegressor()

        tree.fit(X_train_scaled, y_train)
        
        print(stylize(f"Score vertical meters - model: {tree} score: {tree.score(X_test_scaled, y_test)}", colored.fg("yellow")))
        
        profile_na = merged[merged['ProfileScore:'].isna()]
        smaller_ten = merged[merged['ProfileScore:'] <= 10]
        impute_profile = profile_na.append(smaller_ten)
        impute_profile.dropna(subset=['Distance:', 'Vert. meters:'], how='any', inplace=True)

        X_pred = impute_profile[['Distance:', 'Vert. meters:']]

        X_pred_scaled = scaler.transform(X_pred)

        impute_profile['predicted_score'] = tree.predict(X_pred_scaled)
        
        return impute_profile
    
    def parcours_type(df:pd.DataFrame) -> pd.DataFrame:
        
        merged = df.copy()
        
        knn = KNeighborsClassifier(n_neighbors=5)

        px0 = merged[merged['Parcours type:']!='p0']

        px0.dropna(subset = ['ProfileScore:', 'Vert. meters:'], how='any', inplace=True)

        X = px0[['ProfileScore:', 'Vert. meters:']]

        y = px0['Parcours type:']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()

        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        knn.fit(X_train_scaled, y_train)
        
        print(stylize(f"Score vertical meters - model: {knn} score: {knn.score(X_test_scaled, y_test)}", colored.fg("yellow")))

        p0 = merged[merged['Parcours type:']=='p0']

        p0.dropna(subset = ['ProfileScore:', 'Vert. meters:'], how='any', inplace=True)

        X_pred = p0[['ProfileScore:', 'Vert. meters:']]

        X_pred_scaled = scaler.transform(X_pred)

        labels = knn.predict(X_pred_scaled)
        
        p0['adjusted parcours type'] = labels
        
        return p0
    
    def general_preprocess(df:pd.DataFrame) -> pd.DataFrame:
        
            merged = df.copy()
            
            #change parcours names
            parcours = {
            'p1' : 'fl',
            'p2' : 'hi_fl',
            'p3' : 'hi_hi',
            'p4' : 'mo_fl',
            'p5' : 'mo_mo'
            }

            merged['Parcours type:'] = merged['Parcours type:'].map(parcours)
            
            #change to numerical
            parcours_num = {
            'fl':1,
            'hi_fl':2,
            'hi_hi':3,
            'mo_fl':4,
            'mo_mo':5
            }

            merged['parcours_type_num'] = merged['Parcours type:'].map(parcours_num)
            
            #binary for grand tour
            
            grand_tours = ['tour-de-france', 'vuelta-a-espana', 'giro-d-italia']
            
            merged['gt_binary'] = merged['race_name'].map(lambda x: 1 if x in grand_tours else 0)
            
            #add key for DB later
            merged['key'] = merged['name'] + '-' + merged['year'].astype('str') + '-' + merged['race_name']

            def change_bin(x):
                if x == 0:
                    return -1000.
                elif x < 21:
                    return 1.
                else:
                    return 2
                
            merged['result_bin'] = merged['result'].apply(change_bin)
            merged = merged[merged['result_bin'] != -1000.]

            def get_types_bin(x):
                if x == 'etappe':
                    return 0
                else:
                    return 1
                
            def icon_bin(x):
                if x == 'stage':
                    return 0
                else:
                    return 1
                
            merged['types_bin'] = merged['type'].apply(get_types_bin)
            merged['icon_bin'] = merged['icon'].apply(icon_bin)
            
            return merged
    
    #get data
    merged = get_data()
    
    #impute vert_meters + add to df
    vert_na = vert_meters(merged)
    merged.loc[vert_na.index, 'Vert. meters:'] = vert_na['predicted_vert']

    #impute profile + add to df
    impute_profile = profile_score(merged)
    merged.loc[impute_profile.index, 'ProfileScore:'] = impute_profile['predicted_score']
     
    #impute parcours_type + add to df   
    p0 = parcours_type(merged)
    merged.loc[p0.index, 'Parcours type:'] = p0['adjusted parcours type']
    
    merged = general_preprocess(merged)
    
    return merged


if __name__ == '__main__':
    merged = get_data()
    
    preprocessed = preprocess(merged)
    
    print(preprocessed.columns)
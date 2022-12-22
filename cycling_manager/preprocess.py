from importlib import import_module
import pandas as pd
import numpy as np
import datetime
import colored
from colored import stylize
import os
from typing import Tuple

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from colorama import Fore, Style
#comment to disable warnings
pd.options.mode.chained_assignment = None

def get_data(local=True) -> pd.DataFrame:
    
    """
    Function to return fully merged df
    """
    
    if local:
        merged1 = pd.read_csv(f'{os.getenv("LOCAL_DATA_PATH")}/raw_data/merged_clean.csv', index_col=0)
        merged2 = pd.read_csv(f'{os.getenv("LOCAL_DATA_PATH")}/raw_data/merged_clean_2.csv', index_col=0)
        merged3 = pd.read_csv(f'{os.getenv("LOCAL_DATA_PATH")}/raw_data/merged_clean_tdf2022.csv', index_col=0)
        merged4 = pd.read_csv(f'{os.getenv("LOCAL_DATA_PATH")}/raw_data/merged_clean_10s.csv', index_col=0)
        merged5 = pd.read_csv(f'{os.getenv("LOCAL_DATA_PATH")}/raw_data/merged_clean_vuelta2022.csv', index_col=0)
        
        merged = pd.concat([merged1, merged2, merged3, merged4, merged5], ignore_index=True)
        
        merged.drop_duplicates(inplace=True)

        merged.dropna(subset=['ProfileScore:', 'Vert. meters:', 'Distance:'], how='all', inplace=True)
        merged.reset_index(inplace=True, drop=True)
        
        merged['date'] = pd.to_datetime(merged['date'])
        
        print(stylize(f"full df shape {merged.shape}", colored.fg("green")))
        
        return merged
    

def preprocess(df:pd.DataFrame) -> pd.DataFrame:
    
    """
    Preprocess entire df
    """
    
    def vert_meters(df:pd.DataFrame) -> pd.DataFrame:
    #get data
    
        """
        Standard scale + Decistion tree to impute vertical meters
        """
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
        
        """
        Standard scale + Decistion tree to impute profile score
        """
        
        merged = df.copy()
        
        #drop where either profile score or vert meters are missing
        no_na = merged.dropna(subset=['ProfileScore:', 'Vert. meters:'], how='any')

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
        impute_profile = pd.concat([profile_na, smaller_ten])
        impute_profile.dropna(subset=['Distance:', 'Vert. meters:'], how='any', inplace=True)

        X_pred = impute_profile[['Distance:', 'Vert. meters:']]

        X_pred_scaled = scaler.transform(X_pred)

        impute_profile['predicted_score'] = tree.predict(X_pred_scaled)
        
        return impute_profile
    
    def parcours_type(df:pd.DataFrame) -> pd.DataFrame:
        
        """
        Standard scale + Decistion tree to impute parcours type
        """
        
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
        
            """
            preprocess categorical + y_variable
            """
        
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
                    return 0

                
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

def preprocess_features(df : pd.DataFrame) -> pd.DataFrame:

    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Create a scikit-learn preprocessor
        """

        # DISTANCE PIPE
        distance_min = 0.6
        distance_max = 305
        distance_transformer = FunctionTransformer(lambda p: (p - distance_min) /
            (distance_max - distance_min))

        # RESULT PIPE
        result_min = 1
        result_max = 1011
        result_transformer = FunctionTransformer(lambda p: (p - result_min) /
            (result_max - result_min))

        # PROFILE PIPE
        profile_min = 11
        profile_max = 538
        profile_transformer = FunctionTransformer(lambda p: (p - profile_min) /
            (profile_max - profile_min))

        # VERT PIPE
        vert_min = 0
        vert_max = 6104
        vert_transformer = FunctionTransformer(lambda p: (p - vert_min) /
            (vert_max - vert_min))

        # STARTLIST PIPE
        startlist_min = 0
        startlist_max = 1812
        startlist_transformer = FunctionTransformer(lambda p: (p - startlist_min) /
            (startlist_max - startlist_min))

        # STARTLIST PIPE
        speed_min = 0
        speed_max = 82
        speed_transformer = FunctionTransformer(lambda p: (p - speed_min) /
            (speed_max - speed_min))
        
        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
                    [
                        ("distance_transformer", distance_transformer, ["distance"]),
                        ("result_transformer", result_transformer, ["result"]),
                        ("profile_transformer", profile_transformer, ["ProfileScore:"]),
                        ("vert_transformer", vert_transformer, ["Vert. meters:"]),
                        ("startlist_transformer", startlist_transformer, ["Startlist quality score:"]),
                        ("speed_transformer", speed_transformer, ["Avg. speed winner:"]),
                    ], remainder="drop",
                    n_jobs=-1,
                )

        return final_preprocessor
    
    df_small = df[['distance','result', 'ProfileScore:','Vert. meters:', 'Startlist quality score:','Avg. speed winner:']]
    df_big = df.drop(columns=['distance', 'ProfileScore:','Vert. meters:', 'Startlist quality score:','Avg. speed winner:'])
    
    print(Fore.BLUE + "\nPreprocess features..." + Style.RESET_ALL)
    final_preprocessor = create_sklearn_preprocessor()
    
    scaled_df=pd.DataFrame(final_preprocessor.fit_transform(df_small), columns=df_small.columns).rename(columns={'result':'result_scaled'})
    
    return scaled_df.merge(df_big, left_index=True, right_index=True)

def split(df:pd.DataFrame, 
          start: int = 2017,
          end: int = 2022) -> Tuple:
    
    """
    split df into train and test
    """
    
    df_predict = df[df['year'] >= end]
    df_train = df[(df['year'] >= start) & (df['year'] < end)]
    
    #riders to predict
    riders_predict = df_predict[(df_predict['race_name']=='tour-de-france')| (df_predict['race_name']=='vuelta-a-espana')| (df_predict['race_name']=='giro-d-italia')][['name', 'year', 'race_name']]
    riders_predict = riders_predict[riders_predict['year'] == end].drop_duplicates().reset_index(drop=True)
    
    riders = df_train[(df_train['race_name']=='tour-de-france')| (df_train['race_name']=='vuelta-a-espana')| (df_train['race_name']=='giro-d-italia')][['name', 'year', 'race_name']]
    riders = riders[riders['year'] != end].drop_duplicates().reset_index(drop=True).sort_values(by='year', ascending=False)
    
    return riders, riders_predict


if __name__ == '__main__':
    merged = get_data()
    
    preprocessed = preprocess(merged)
    
    split(preprocessed)
    
    print(preprocessed.columns)
from bs4 import BeautifulSoup
import requests
import pandas as pd
import itertools
import numpy as np
import colored
from colored import stylize

def scrape_participants(tour:str, year:int) -> list:
    
    ''' Function to scrape all participants of tour in a given year'''
    
    # define url for startlist
    url = f'https://www.procyclingstats.com/race/{tour}/{year}/gc/startlist'
    print(url)
    print(stylize(f"scrape participants", colored.fg("blue")))
    
    #scrape page
    response = requests.get(url).content
    soup = BeautifulSoup(response, "html.parser")
    
    #get all_teams
    all_teams = soup.find_all('li', class_='team')
            
    #loop over teams
    master_ls = []
    for t in all_teams:
        team = t.a.text
        riders = t.find_all('a', class_='blue')
        for r in riders:
            dict = {}
            href = r['href']
            dict['rider'] = href.split('/')[-1]
            dict['team'] = team
            dict['href'] = href
            dict['tour'] = tour
            dict['year'] = int(year)
            master_ls.append(dict)
    
    return master_ls

def scrape_performance(rider:str, endpoint:str, year:int) -> list:
    
    """Function to scrape performance of participant in tour"""
    
    #set up
    print(stylize(f"scrape performance", colored.fg("red")))
    
    base_url = 'https://www.procyclingstats.com/'
    url = base_url+endpoint+'/'+str(year)
    
    response = requests.get(url).content
    soup = BeautifulSoup(response)
    
    result_ls = []
    
    #get stage_race results
    stage_races = soup.find_all('tr', {'data-main': '0'})
    
    for o in stage_races:
        dict = {}
        o = o.find_all('td')
        dict['name'] = rider
        dict['year'] = str(year)
        dict['type'] = 'etappe'
        dict['date'] = o[0].text
        if len(dict['date']) == 0:
            dict['type'] = 'gc'
        dict['result'] = o[1].text
        dict['gc'] = o[2].text
        try:
            dict['icon'] = o[3].find('span', class_='icon')['class'][-1]
        except TypeError:
            dict['icon'] = 'stage'
        dict['race_ref'] = o[4].a['href']
        dict['race_name'] = dict['race_ref'].split('/')[1]
        dict['race_detail'] = o[4].a.text
        try:
            dict['race_rank'] = o[4].a.span.text
        except AttributeError:
            dict['race_rank'] = o[4].a.span
        dict['distance'] = o[5].text
        result_ls.append(dict)
    
    #get one day race results
    one_day_races = soup.find_all('tr', {'data-main': '1'})
    
    for o in one_day_races:
        dict = {}
        o = o.find_all('td')
        dict['name'] = rider
        dict['year'] = str(year)
        dict['type'] = 'one_day'
        dict['date'] = o[0].text
        dict['result'] = o[1].text
        dict['gc'] = o[2].text
        try:
            dict['icon'] = o[3].find('span', class_='icon')['class'][-1]
        except TypeError:
            dict['icon'] = 'stage'
        dict['race_ref'] = o[4].a['href']
        dict['race_name'] = dict['race_ref'].split('/')[1]
        dict['race_detail'] = o[4].a.text
        try:
            dict['race_rank'] = o[4].a.span.text
        except AttributeError:
            dict['race_rank'] = o[4].a.span
        dict['distance'] = o[5].text
        result_ls.append(dict)
    
    return result_ls

def clean_df(ls:list) -> pd.DataFrame:
    
    """Function to clean performance DF"""
    
    print(stylize(f"clean df", colored.fg("yellow")))
    
    stage_s = list(np.arange(2,32,2))+list(np.arange(32,48,4))+[50]
    stage_s_i = list(np.arange(1,21,1))
    stage_s_dict = dict(zip(stage_s_i, stage_s[::-1]))
    
    df = pd.DataFrame(ls)
    
    index_drop = df[df['result']==''].index

    dropped_df = df.drop(index_drop)

    index_drop = dropped_df[dropped_df['type']=='gc'].index

    dropped_df = dropped_df.drop(index_drop)
    
    dropped_df['date'] = pd.to_datetime(dropped_df['date'] + '.' + dropped_df['year'], infer_datetime_format=True)
    
    dropped_df['result'] =  dropped_df['result'].replace('DNF', 0).replace('DNS', 0).replace('OTL', 0).replace('DSQ', 0).replace('DF', 0).astype('int')
    
    dropped_df['points'] = dropped_df['result'].map(stage_s_dict).fillna('0').astype('int')
    
    return dropped_df

def get_profile(ls:list, clean_df:pd.DataFrame) -> list:
    extra_info_ls = []
    i=0
    
    print(stylize(f"get profile", colored.fg("green")))
    
    for ref in ls:
        print(i/len(clean_df.race_ref.unique()))
        #create url
        base_url = 'https://www.procyclingstats.com/'
        url = base_url + ref
        response = requests.get(url).content
        soup = BeautifulSoup(response)
        
        print(url)
        
        #get al info
        dict = {}
        try:
            stage = soup.find('ul', class_='infolist').find_all('li')
                
            dict['href'] = ref
            #get speed
            try:
                dict[stage[2].find_all('div')[0].text] = float(stage[2].find_all('div')[1].text.strip(' km/h'))
            except ValueError:
                dict[stage[2].find_all('div')[0].text] = np.nan
            #get distance
            try:
                dict[stage[4].find_all('div')[0].text.strip()] = float(stage[4].find_all('div')[1].text.strip(' km'))
            except ValueError:
                dict[stage[4].find_all('div')[0].text.strip()] = np.nan
            #get parcours type
            try:
                dict[stage[6].find_all('div')[0].text.strip()] = stage[6].find_all('div')[1].span['class'][-1]
            except (ValueError, TypeError):
                dict[stage[6].find_all('div')[0].text.strip()] = np.nan
            #get profile score
            try:
                dict[stage[7].find_all('div')[0].text.strip()] = int(stage[7].find_all('div')[1].text)
            except ValueError:
                dict['ProfileScore:'] = np.nan
            #get vert meters
            try:
                dict[stage[8].find_all('div')[0].text.strip()] = int(stage[8].find_all('div')[1].text)
            except (ValueError, IndexError):
                dict['Vert. meters:'] = np.nan
            #get startlist
            try:
                dict[stage[12].find_all('div')[0].text.strip()] = int(stage[12].find_all('div')[1].text)
            except (ValueError, IndexError):
                dict['Startlist quality score:'] = np.nan
            #get won how
            try:
                dict[stage[13].find_all('div')[0].text]= stage[13].find_all('div')[1].text
            except (ValueError, IndexError):
                dict['Won how:'] = np.nan
            
            extra_info_ls.append(dict)
            
            i += 1
        
        except AttributeError:
            print(ref)
            
    return extra_info_ls

def get_tour_stages(tour :str) -> list:
    new_ls = []

    for i in range(1,22):
        new_ls.append(f'race/{tour}/2022/stage-{i}')
        
    return new_ls


def run_scraper(tours:list, years:list)-> pd.DataFrame:
    
    """Function to run whole scraper"""
    
    #get list of participants for every year/tour
    participants_ls = []
    for y, t in list(itertools.product(years, tours)):
        participants_ls.append(scrape_participants(t, y))
    
    participants_df = pd.DataFrame(list(itertools.chain(*participants_ls)))
    
    # get performance every participant
    performance_ls = []
    for index, row in participants_df.iterrows():
        performance_ls.append(scrape_performance(row['rider'], row['href'], row['year']))
    
    #performance_df = pd.DataFrame(list(itertools.chain(*performance_ls)))
    
    #clean performance df
    performance_clean = clean_df(list(itertools.chain(*performance_ls)))
    
    performance_clean.drop_duplicates(inplace=True)
    
    #tour_profile_df = pd.DataFrame(get_profile(get_tour_stages(tours)))
    
    past_profile_df = pd.DataFrame(get_profile(performance_clean.race_ref.unique())).rename(columns={'href':'race_ref'})
    
    merged = performance_clean.merge(past_profile_df, on='race_ref')
    merged['points'] = merged['points'].astype('float')
    merged['adjusted_points'] = merged['points'] * merged['ProfileScore:']  * merged['Startlist quality score:']    
    
    return merged
    

if __name__ == '__main__':
    print(run_scraper(['tour-de-france'], [2022]))
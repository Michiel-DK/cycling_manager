from bs4 import BeautifulSoup
import requests
import pandas as pd
import itertools
import numpy as np
import colored
from colored import stylize

def scrape_participants(tour, year):
    # define url for startlist
    url = f'https://www.procyclingstats.com/race/{tour}/{year}/stage-21/startlist'
    print(stylize(f"Getting startlist for {tour} - {year}", colored.fg("blue")))
    
    #scrape page
    response = requests.get(url).content
    soup = BeautifulSoup(response)
    
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

if __name__ == '__main__':
    print(scrape_participants('tour-de-france', 2022))
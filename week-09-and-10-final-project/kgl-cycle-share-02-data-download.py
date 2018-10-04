## ######################################################################### ##
## Analysis of 
## For EdX Course
## Python for Data Science (Week 9 and 10 Final Project)
## ######################################################################### ##

## ========================================================================= ## 
## import libraries
## ========================================================================= ##

import requests
import io
import zipfile
import os
import urllib.parse
import re   ## for regular expressions
from itertools import chain  ## for chain, similar to R's unlist (flatten lists)
import collections   ## for Counters (used in frequency tables, for example)
import numpy as np
import pandas as pd

## ========================================================================= ##
## download data
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## montreal bike share data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## BIXI Montreal (public bicycle sharing system)
## Data on North America's first large-scale bike sharing system
## https://www.kaggle.com/aubertsigouin/biximtl/home

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## weather data from canadian government
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## station id available from:
## ftp://ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/
## ftp://ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv

## define station, years and month:
## (for downloading and data prep later)
## see setup.py file

for year in year_list:
    for month in month_list:
        ## define url for bulk data download of http://climate.weather.gc.ca/climate_data :
        url_string = 'http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={0}&Year={1}&Month={2}&Day=14&timeframe=1&submit=Download+Data'.format(station_id, year, month)
        
        ## define filename, as well as target path:
        filename_target = 'weather_montreal_{0}_{1:04d}_{2:02d}.csv'.format(station_id, year, month)
        path_dat = './data'
        
        ## check if file exists, only download if it does not exist:
        if os.path.isfile(os.path.join(path_dat, filename_target)):
            print('year {0:04d}, month {1:02d} -- file exists, skipped.'.format(year, month))
        else:
            ## download (commented out in order not to repeat it every time):
            r = requests.get(url_string, allow_redirects=True)
            open(os.path.join(path_dat, filename_target), 'wb').write(r.content)

            ## display progress:
            print('year {0:04d}, month {1:02d} -- downloaded.'.format(year, month))

print('data download complete.')

# ## check download:
# os.getcwd()
# os.listdir()




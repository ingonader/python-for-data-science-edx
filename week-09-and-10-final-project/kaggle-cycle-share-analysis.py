## ######################################################################### ##
## Analysis of 
## For EdX Course
## Python for Data Science (Week 9 and 10 Final Project)
## ######################################################################### ##

## questions:
## * syntax completion in editor of jupyterlab?
## * ...
## * [[todo]]: check out [[todo]] and [[?]] in other files

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
from pandas.api.types import CategoricalDtype  ## for sorted plotnine/ggplot categories
from plotnine import *
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import folium

%matplotlib osx

## ========================================================================= ##
## global variables and options
## ========================================================================= ##

path_dat = './data'

pd.set_option('display.max_columns', 50)

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

## http://climate.weather.gc.ca/climate_data/hourly_data_e.html?timeframe=1&Year=2018&Month=9&Day=30&hlyRange=2013-02-13%7C2018-09-30&dlyRange=2013-02-14%7C2018-09-30&mlyRange=%7C&StationID=51157&Prov=QC&urlExtension=_e.html&searchType=stnProx&optLimit=yearRange&StartYear=2014&EndYear=2017&selRowPerPage=25&Line=1&txtRadius=25&optProxType=city&selCity=45%7C31%7C73%7C39%7CMontr%C3%A9al&selPark=
## ftp://ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/Readme.txt
# for year in `seq 2014 2017`;do for month in `seq 1 12`;do wget --content-disposition "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=1706&Year=${year}&Month=${month}&Day=14&timeframe=1&submit= Download+Data" ;done;done


## station id available from:
## ftp://ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/
## ftp://ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv
# station_id = 51157 # MONTREAL INTL A; QUEBEC 
station_id = 10761 # "MCTAVISH", "QUEBEC"
year_list = [2014, 2015, 2016, 2017]
month_list = list(range(1, 13))

for year in year_list:
    for month in month_list:
        ## display progress:
        print('year {0:04d}, month {1:02d}'.format(year, month))
        ## define url for bulk data download of http://climate.weather.gc.ca/climate_data :
        url_string = 'http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={0}&Year={1}&Month={2}&Day=14&timeframe=1&submit=Download+Data'.format(station_id, year, month)
        
        ## define filename, as well as target path:
        filename_target = 'weather_montreal_{0}_{1:04d}_{2:02d}.csv'.format(station_id, year, month)
        path_dat = './data'

        ## download (commented out in order not to repeat it every time):
        r = requests.get(url_string, allow_redirects=True)
        open(os.path.join(path_dat, filename_target), 'wb').write(r.content)

# ## check download:
# os.getcwd()
# os.listdir()

## ========================================================================= ##
## Load data files
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## montreal bike share data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## inspect trip data:
tmp = pd.read_csv('./data/OD_2014.csv').head()
tmp.head()
tmp.dtypes

## inspect stations data:
pd.read_csv('./data/Stations_2014.csv').head()
pd.read_csv('./data/Stations_2015.csv').head()
pd.read_csv('./data/Stations_2016.csv').head()
pd.read_csv('./data/Stations_2017.csv').head()
pd.read_csv('./data/Stations_2014.csv').head().dtypes

## data for these years:
dat_years = [2014, 2015, 2016, 2017]

## function to read each file in the same way:
def read_trip_csv(path_dat, year, filename_prefix = 'OD_', filename_suffix = '.csv'):
    dtype = {'id': 'int64',
        'start_date' : 'str',
        'start_station_code' : 'str', 
         'end_date' : 'str', 
         'end_station_code' : 'str', 
         'duration_sec' : 'int64', 
         'is_member' : 'int64'}
    names = list(dtype.keys())
    parse_dates = ['start_date', 'end_date']
    filename = filename_prefix + '{0}'.format(year) + filename_suffix
    dat = pd.read_csv(os.path.join(path_dat, filename), names = names, skiprows = 1, 
                      dtype = dtype, parse_dates = parse_dates)
    return dat

#read_trip_csv(path_dat, 2014)

## function to read each Stations_ file in the same way:
def read_stations_csv(path_dat, year, filename_prefix = 'Stations_', filename_suffix = '.csv'):
    dtype = {'code' : 'str', 
             'name' : 'str',
             'latitude' : 'float',
             'longitude' : 'float'}
    ## new column in 2017:
    if (year == 2017):
        dtype['is_public'] = 'int'
    names = list(dtype.keys())
    filename = filename_prefix + '{0}'.format(year) + filename_suffix
    dat = pd.read_csv(os.path.join(path_dat, filename), names = names, skiprows = 1, 
                      dtype = dtype)
    ## add is_public column if not available in data:
    if (year < 2017):
        dat['is_public'] = None
    dat['year'] = year
    return dat         
    
#read_stations_csv(path_dat, 2014)
#read_stations_csv(path_dat, 2017)

## read all files into a list for trips and stations:
dat_trip_raw_list =     [read_trip_csv(path_dat, i)     for i in dat_years]
dat_stations_raw_list = [read_stations_csv(path_dat, i) for i in dat_years]

## make one pandas dataframe from lists (for trips and stations):
dat_trip_raw =     pd.concat(dat_trip_raw_list, axis = 0)
dat_stations_raw = pd.concat(dat_stations_raw_list, axis = 0)

## basic inspection of raw trip data:
dat_trip_raw.info()
dat_trip_raw.shape   # (14598961, 7)

type(dat_trip_raw)   # pandas.core.frame.DataFrame

## get column names:
list(dat_trip_raw.columns.values)      
list(dat_trip_raw)                     ## same

## lots of info, but lengthy:
dat_trip_raw.__dict__                  ## lengthy, equivalent to vars(<>)
vars(dat_trip_raw)                     ## lengthy, equivalent to <>.__dict__

## summary of data frame
dat_trip_raw.describe().round(3)        ## similar to R's summary()

## list all methods:
dir(dat_trip_raw)                      ## list all methods?
list(filter(lambda x: re.search(r'unsta', x), dir(dat_trip_raw)))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## weather data from canadian government
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## get column types?
## inspect weather data:
tmp = pd.read_csv(os.path.join(path_dat, 'weather_montreal_10761_2014_01.csv'), skiprows = 15).head()
tmp = pd.read_csv(os.path.join(path_dat, 'weather_montreal_10761_2015_01.csv'), skiprows = 15).head()
tmp.head()
tmp.dtypes

# year = 2017
# month = 12
# filename_prefix = 'weather_montreal_'
# filename_suffix = '.csv'

## function to read weather data csv
def read_weather_csv(path_dat, year, month, station_id,
                     cols = ['Date/Time', 'Year', 'Month', 'Temp (°C)', 
                             'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)',
                             'Wind Spd (km/h)', 'Stn Press (kPa)', 'Wind Chill'],
                     filename_prefix = 'weather_montreal_', 
                     filename_suffix = '.csv'):
    dtype = {'Date/Time' :                'str',
         'Year' :                    'int64',
         'Month' :                   'int64',
         'Day' :                     'int64',
         'Time' :                    'str',
         'Temp (°C)' :               'float64',
         'Temp Flag' :               'str',
         'Dew Point Temp (°C)' :     'float64',
         'Dew Point Temp Flag' :     'str',
         'Rel Hum (%)' :             'float64',
         'Rel Hum Flag' :            'str',
         'Wind Dir (10s deg)' :      'float64',
         'Wind Dir Flag' :           'str',
         'Wind Spd (km/h)' :         'float64',
         'Wind Spd Flag' :           'str',
         'Visibility (km)' :         'float64',
         'Visibility Flag' :         'str',
         'Stn Press (kPa)' :         'float64',
         'Stn Press Flag' :           'str',
         'Hmdx' :                    'float64',
         'Hmdx Flag' :               'str',
         'Wind Chill' :              'float64',
         'Wind Chill Flag' :         'str',
         'Weather' :                 'str'}
    names = list(dtype.keys())
    parse_dates = ['Date/Time']
    filename = filename_prefix + '{0}_{1:04d}_{2:02d}'.format(station_id, year, month) + filename_suffix
    dat = pd.read_csv(os.path.join(path_dat, filename), skiprows = 16, dtype = dtype, names = names, parse_dates = parse_dates)
    return dat[cols]

#read_weather_csv(path_dat, 2017, 1, station_id).head()
#read_weather_csv(path_dat, 2015, 1, station_id).head()

## read each file into a list:
dat_weather_raw_list =  [read_weather_csv(path_dat, i, j, station_id) \
                        for i in year_list \
                        for j in month_list]

# for i in year_list:
#    for j in month_list:
#        tmp = read_weather_csv(path_dat, i, j, station_id)

## combine list into single dataframe:
dat_weather_raw =     pd.concat(dat_weather_raw_list, axis = 0)

dat_weather_raw.info()

## ========================================================================= ##
## exploratory analysis of stations
## ========================================================================= ##

weatherstation_latlon = [[45.5047416666667, -73.5791666666667], 
[45.4705555555556, -73.7408333333333], 
[45.4677777777778, -73.7416666666667], 
[45.5, -73.85], 
[45.5175, -73.4169444444445], 
[45.5166666666667, -73.4166666666667], 
[45.3833333333333, -73.4333333333333], 
[45.7, -73.5], 
[45.4272222222222, -73.9291666666667]]

weatherstation_name = ['MCTAVISH', 
'MONTREAL INTL A', 
'MONTREAL/PIERRE ELLIOTT TRUDEAU INTL', 
'STE GENEVIEVE', 
'MONTREAL/ST-HUBERT', 
'MONTREAL/ST-HUBERT A', 
'LAPRAIRIE', 
'RIVIERE DES PRAIRIES', 
'STE-ANNE-DE-BELLEVUE 1']

## basic map:
folium_map = folium.Map(location = loc, zoom_start = 11)
## add markers for bike stations:
for i in range(0, len(dat_stations_raw)):
    folium.CircleMarker(location = list(dat_stations_raw[['latitude', 'longitude']].iloc[i]),
                        radius = 2)\
    .add_to(folium_map)
## add markers for possible weather stations:
for i in range(0, len(weatherstation_name)):
    folium.Marker(location = weatherstation_latlon[i],
                  popup = weatherstation_name[i])\
    .add_to(folium_map)
## save plot as html:
folium_map.save("map-of-bike-and-possible-weather-stations.html")

## ========================================================================= ##
## exploratory analysis of trips
## ========================================================================= ##

## most common starting stations:
start_station_counter = collections.Counter(dat_trip_raw['start_station_code'])
start_station_counter.most_common(10)

## make pandas data frame for visualizing with ggplot/plotnine:
dat_start_station_freq = pd.DataFrame(
    start_station_counter.most_common(20),
    columns = ['start_station_code', 'frequency'])
dat_start_station_freq.rename(index = dat_start_station_freq['start_station_code'], inplace = True)

## frequency series (for sorting):
## (pandas series with index that corresponds to categories):
dat_start_station_freq['frequency']

## create list for sorting:
#station_list = dat_start_station_freq['start_station_code'].value_counts().index.tolist()
station_list = dat_start_station_freq['frequency'].index.tolist()
station_cat = CategoricalDtype(categories=station_list, ordered=True)
dat_start_station_freq['start_station_code_cat'] = \
    dat_start_station_freq['start_station_code'].astype(str).astype(station_cat)

## plot counter data (frequency table, with identity relation):
## (sorting does not work here)
ggplot(dat_start_station_freq, aes(x = 'start_station_code_cat', y = 'frequency')) + \
    geom_bar(stat = 'identity') + \
    coord_flip()

## total number of trips:
dat_trip_raw.shape[0]  #  14 598 961 (14 Mio)

## [[here]]


## number of trips per day:
dat_trip_raw.resample() ## [[?]]
dat_trip_raw.head()
## number of trips per hour:

## average length of trips:
%matplotlib inline
sample_frac = .01
ggplot(dat_trip_raw.sample(frac = sample_frac), 
       aes(x = 'duration_sec')) + geom_histogram(bins = 100)


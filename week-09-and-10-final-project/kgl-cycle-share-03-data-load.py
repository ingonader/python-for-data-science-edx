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
## Load data files
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## montreal bike share data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## inspect trip data:
tmp = pd.read_csv(os.path.join(path_dat, 'OD_2014.csv')).head()
tmp.head()
tmp.dtypes

## inspect stations data:
pd.read_csv(os.path.join(path_dat, 'Stations_2014.csv')).head()
pd.read_csv(os.path.join(path_dat, 'Stations_2015.csv')).head()
pd.read_csv(os.path.join(path_dat, 'Stations_2016.csv')).head()
pd.read_csv(os.path.join(path_dat, 'Stations_2017.csv')).head()
pd.read_csv(os.path.join(path_dat, 'Stations_2014.csv')).head().dtypes

# ## data for these years:
# dat_years = [2014, 2015, 2016, 2017]
# ## see setup.py

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


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## write raw data to disk using feather
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

import feather

feather.write_dataframe(
    dat_weather_raw,
    os.path.join(path_dat, 'dat_weather_raw.feather')
)



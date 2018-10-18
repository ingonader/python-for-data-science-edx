## ######################################################################### ##
## Analysis of 
## For EdX Course
## Python for Data Science (Week 9 and 10 Final Project)
## ######################################################################### ##

## questions:
## * syntax completion in editor of jupyterlab?
## * ...
## * [[todo]]: check out [[todo]] and [[?]] in other files

## [[todo]]
## * clean up file (remove unnecessary code / comments)


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
import seaborn as sns  ## for correlation heatmap
#from mpl_toolkits.basemap import Basemap
import folium

## ========================================================================= ##
## global variables and options
## ========================================================================= ##

path_dat = './data'
path_out = './output'

pd.set_option('display.max_columns', 50)

## weather data: station_id, year_list, month_list for data download and data prep:

# station_id = 51157 # MONTREAL INTL A; QUEBEC 
station_id = 10761 # "MCTAVISH", "QUEBEC"
year_list = [2014, 2015, 2016, 2017]
month_list = list(range(1, 13))

## cycle trip data:

## data for these years:
dat_years = [2014, 2015, 2016, 2017]

## "translation" from "quoted" to long variable / feature names:
varnames_long_dict = {
    "Q('Month')" :              "Month (1-12)",
    "Q('Temp (°C)')" :          "Temperature (°C)",    
    "Q('Rel Hum (%)')" :        "Relative Humidity (%)",      
    "Q('Wind Dir (10s deg)')" : "Wind Direction (deg)",             
    "Q('Wind Spd (km/h)')" :    "Wind Speed (km/h)",          
    "Q('Stn Press (kPa)')" :    "Atmospheric Pressure (kPa)",          
    "Q('hr_of_day')" :          "Hour of the Day (0-23)",    
    "Q('day_of_week')" :         "Day of the Week (0-6)"
}

## "translation" of unquoted to long variable / feature names:
varnames_orig_long_dict = {
    'trip_cnt' 				: 'Trip Count',
    'Year' 					: 'Year (2014-2017)',
    'Month' 				: 'Month (1-12)',
    'Temp (°C)' 			: 'Temperature (°C)',
    'Dew Point Temp (°C)' 	: 'Dew Point (°C)',
    'Rel Hum (%)' 			: 'Relative Humidity (%)',
    'Wind Dir (10s deg)' 	: 'Wind Direction (10s deg)',
    'Wind Spd (km/h)'  		: 'Wind Speed (km/h)',
    'Stn Press (kPa)' 		: 'Atmospheric Pressure (kPa)',
    'hr_of_day' 			: 'Hour of the Day (0-23)',
    'day_of_week' 			: 'Day of the Week (0-6)'
}





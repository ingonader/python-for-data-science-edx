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
from pandas.api.types import CategoricalDtype  ## for sorted plotnine/ggplot categories
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns  ## for correlation heatmap
#from mpl_toolkits.basemap import Basemap
import folium


## ========================================================================= ##
## data preparation
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## aggregate trip data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## set time index for dataframe (in order to use `resample`):
dat_trip_raw.set_index(
    pd.DatetimeIndex(dat_trip_raw['start_date']), 
    inplace = True)  ## [[?]] use inplace = True?

dat_trip_raw.columns

## daily summary of trips:
dat_trip_day = pd.DataFrame()
dat_trip_day['trip_cnt'] = dat_trip_raw['start_date'].resample('24h').count()
dat_trip_day['duration_min_mean'] = dat_trip_raw['duration_sec'].resample('24h').mean() / 60
dat_trip_day['start_date'] = dat_trip_day.index
dat_trip_day.head()
dat_trip_day.shape

## hourly summary of trips:
dat_trip_hr = pd.DataFrame()
dat_trip_hr['trip_cnt'] = dat_trip_raw['start_date'].resample('1h').count()
dat_trip_hr['duration_min_mean'] = dat_trip_raw['duration_sec'].resample('1h').mean() / 60
dat_trip_hr['start_date'] = dat_trip_hr.index
dat_trip_hr.head()
dat_trip_hr.shape

## [[to do]]
## * exclude rows with zero trips [[?]]
## * make two models?
##   * one for predicting now rides vs. some rides, 
##   * and one for number of rides?
## * or just leave it and use random forest (and no regression-based model)?

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## join trip data to weather data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

dat_hr_all = pd.merge(
    left = dat_trip_hr,
    right = dat_weather_raw,
    how = 'left',
    left_on = 'start_date',
    right_on = 'Date/Time')

dat_hr_all.head()
dat_hr_all.info()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## add further columns
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## add hour of day:
dat_hr_all['hr_of_day'] = dat_hr_all['start_date'].dt.hour

## add day of week:
dat_hr_all['day_of_week'] = dat_hr_all['start_date'].dt.weekday ## or weekday or weekday_name

## add trip indicator:
dat_hr_all['trip_ind'] = (dat_hr_all['trip_cnt'] > 0).astype(int)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## further data prep
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

dat_hr_all.info()

## don't use wind chill, as too many missings: (only 6713 out of 30360)
del dat_hr_all['Wind Chill']

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## missing value imputation
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## cross-table of trip_cnt and duratin_min_mean:
pd.crosstab(dat_hr_all['trip_cnt'] == 0, 
            pd.isnull(dat_hr_all['duration_min_mean']),
            rownames = ['trip_cnt == 0'],
            colnames = ['isnull(duration_min_mean)'])


## [[?]] impute missing trip durations with zero?

## [[todo]]

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## write raw data to disk using feather
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

import feather

feather.write_dataframe(
    dat_hr_all,
    os.path.join(path_dat, 'dat_hr_all.feather')
)







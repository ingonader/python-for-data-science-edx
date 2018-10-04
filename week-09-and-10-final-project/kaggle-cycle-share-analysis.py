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

## station id available from:
## ftp://ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/
## ftp://ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv

## define station, years and month:
## (for downloading and data prep later)

# station_id = 51157 # MONTREAL INTL A; QUEBEC 
station_id = 10761 # "MCTAVISH", "QUEBEC"
year_list = [2014, 2015, 2016, 2017]
month_list = list(range(1, 13))

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

loc = list(dat_stations_raw[['latitude', 'longitude']].mean())

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

## [[todo]]
## * change color of markers?
## * [[?]] make plot with other technique in python? how? basemap? how to get city streets?

## ========================================================================= ##
## exploratory analysis of trips
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## generic data exploration
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

%matplotlib osx
dat_hr_all.hist(bins = 20, figsize = (18, 9))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## most common starting stations
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

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

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## total number of trips
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## total number of trips:
dat_trip_raw.shape[0]  #  14 598 961 (14 Mio)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## number of trips
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## histogram of number of trips per hour:
ggplot(dat_trip_hr, aes(x = 'trip_cnt')) + geom_histogram(bins = 100)
ggplot(dat_trip_hr[dat_trip_hr['trip_cnt'] > 0], 
       aes(x = 'trip_cnt')) + geom_histogram(bins = 100)

dat_trip_hr[['trip_cnt']].describe()

## histogram of number of trips per day:
ggplot(dat_trip_day, aes(x = 'trip_cnt')) + geom_histogram(bins = 100)
ggplot(dat_trip_day[dat_trip_day['trip_cnt'] > 0], 
       aes(x = 'trip_cnt')) + geom_histogram(bins = 100)

dat_trip_day[['trip_cnt']].describe()


## line plot of number of trips per hour:
ggplot(dat_trip_hr, aes(y = 'trip_cnt', x = 'start_date')) + \
    geom_point(alpha = .05) + \
    geom_smooth(method = 'mavg', method_args = {'window' : 14*24}, 
                color = 'red', se = False)

## line plot of number of trips per day:
ggplot(dat_trip_day, aes(y = 'trip_cnt', x = 'start_date')) + \
    geom_point(alpha = .1) + \
    geom_smooth(method = 'mavg', method_args = {'window' : 14}, 
                color = 'red', se = False)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## average length of trips
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## average length of trips (raw data):
%matplotlib inline
sample_frac = .01
ggplot(dat_trip_raw.sample(frac = sample_frac), 
       aes(x = 'duration_sec')) + geom_histogram(bins = 100)

## average length of trips (hourly summary data):
%matplotlib inline
ggplot(dat_trip_hr, 
       aes(x = 'duration_min_mean')) + geom_histogram(bins = 100)

## average length of trips (daily summary data):
%matplotlib inline
ggplot(dat_trip_day, 
       aes(x = 'duration_min_mean')) + geom_histogram(bins = 100)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## weather data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

dat_weather_raw.describe()
dat_weather_raw.info()
dat_weather_raw.head()

## line plot of number of some weather metrics

## select which metrics to plot (or not plot):
wch_cols = list(set(dat_weather_raw.columns) - set(['Date/Time', 'Year', 'Month']))

for i in wch_cols:
    p = ggplot(dat_weather_raw, aes(y = i, x = 'Date/Time')) + \
        geom_point(alpha = .05) + \
        geom_smooth(method = 'mavg', method_args = {'window' : 14*24}, 
                color = 'red', se = False)
    print(p)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## multivariate data plots
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

dat_hr_all.info()

## exploring trip_cnt relationships:
%matplotlib inline
ggplot(dat_hr_all, aes(y = 'trip_cnt', x = 'Temp (°C)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'trip_cnt', x = 'Wind Spd (km/h)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'trip_cnt', x = 'Dew Point Temp (°C)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'trip_cnt', x = 'Rel Hum (%)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'trip_cnt', x = 'Wind Dir (10s deg)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'trip_cnt', x = 'Stn Press (kPa)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)


## exploring duration_min_mean relationships:
%matplotlib inline
ggplot(dat_hr_all, aes(y = 'duration_min_mean', x = 'Temp (°C)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'duration_min_mean', x = 'Wind Spd (km/h)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'duration_min_mean', x = 'Dew Point Temp (°C)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'duration_min_mean', x = 'Rel Hum (%)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'duration_min_mean', x = 'Wind Dir (10s deg)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

ggplot(dat_hr_all, aes(y = 'duration_min_mean', x = 'Stn Press (kPa)', color = 'hr_of_day')) + \
    geom_point(alpha = .1)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## check correlations
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## correlation:
dat_hr_all.corr()
## [[todo]] correlations!

## correlation heatmap:

## simple:
cormat = dat_hr_all.corr()
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(cormat, cmap = colormap)

## more complex:
%matplotlib inline
#%matplotlib osx
fig, ax = plt.subplots(figsize = (10, 10))
#Generate Color Map, red & blue
colormap = sns.diverging_palette(220, 10, as_cmap = True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(cormat, cmap = colormap, annot = True, fmt = ".2f", center = 0)
#Apply xticks
#plt.xticks(range(len(cormat.columns)), cormat.columns);
#Apply yticks
#plt.yticks(range(len(cormat.columns)), cormat.columns)
#show plot
plt.show()

## ========================================================================= ##
## modeling number of trips
## ========================================================================= ##

## using all data (as opposed to using only data with only non-zero trips):

import patsy ## for design matrices like R
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

## ------------------------------------------------------------------------- ##
## define features and formula
## ------------------------------------------------------------------------- ##

## convert categorical variables to strings
## (in order for patsy to automatically dummy-code them without
## having to use the C() function):

# dat_hr_all['Month'] = dat_hr_all['Month'].astype('str')
# dat_hr_all['hr_of_day'] = dat_hr_all['hr_of_day'].astype('str')

## interesting:
## accuracy seems to be higher for non-categorical features!
## categorical     r^2 (train/test):  0.82382485701064379 / 0.79690027372546179
## non-categorical r^2 (train/test):  0.85217150610946379 / 0.82428144266270897

## also for weekday_name vs. weekday
## categorical     r^2 (train/test):  0.89759620156755338 / 0.87826354433724219
## non-categorical r^2 (train/test):  0.90785676507150148 / 0.89120320733183955

## define target and features:
target = 'trip_cnt'
features = ['Month',
            'Temp (°C)',
            # 'Dew Point Temp (°C)', ## -- exclude, because highly correlated with Temp
            'Rel Hum (%)',
            'Wind Dir (10s deg)',
            'Wind Spd (km/h)',
            'Stn Press (kPa)',
            'hr_of_day',
            'day_of_week']
list(dat_hr_all)

## add patsy-quoting to features (for weird column names):
target = 'Q(\'' + target + '\')' 
features = ['Q(\'' + i + '\')' for i in features]

## formula as text for patsy: without interactions
formula_txt = target + ' ~ ' + \
    ' + '.join(features) + ' - 1'
formula_txt

## create design matrices using patsy (could directly be used for modeling):
#patsy.dmatrix?
dat_y, dat_x = patsy.dmatrices(formula_txt, dat_hr_all, 
                               NA_action = 'drop',
                               return_type = 'dataframe')
dat_x.head()

## other possibilities for dummy coding:
## * pd.get_dummies [[?]] which to use?

## ------------------------------------------------------------------------- ##
## train / test split
## ------------------------------------------------------------------------- ##

## Split the data into training/testing sets (using patsy/dmatrices):
dat_train_x, dat_test_x, dat_train_y, dat_test_y = train_test_split(
    dat_x, dat_y, test_size = 0.33, random_state = 142)

## convert y's to Series (to match data types between patsy and non-patsy data prep:)
dat_train_y = dat_train_y[target]
dat_test_y = dat_test_y[target]

## ------------------------------------------------------------------------- ##
## normalize data
## ------------------------------------------------------------------------- ##

## [[todo]]


## ------------------------------------------------------------------------- ##
## estimate model and evaluate fit and model assumptions
## ------------------------------------------------------------------------- ##

## Instantiate random forest estimator:
mod_rf = RandomForestRegressor(n_estimators = 500, 
                               random_state = 42,
                               max_depth = 20, 
                               min_samples_split = 50,
                               min_samples_leaf = 20,
                               oob_score = True,
                               n_jobs = -2,
                               verbose = 1)

## Train the model using the training sets:
mod_rf.fit(dat_train_x, dat_train_y)

## [[?]] missing: how to plot oob error by number of trees, like in R?
    
## Make predictions using the testing set
dat_test_pred = mod_rf.predict(dat_test_x)
dat_train_pred = mod_rf.predict(dat_train_x)

## Inspect model:
mean_squared_error(dat_train_y, dat_train_pred)  # MSE in training set
mean_squared_error(dat_test_y, dat_test_pred)    # MSE in test set
r2_score(dat_train_y, dat_train_pred)            # R^2 (r squared) in test set
r2_score(dat_test_y, dat_test_pred)              # R^2 (r squared) in test set

## ------------------------------------------------------------------------- ##
## variable importance
## ------------------------------------------------------------------------- ##

## variable importance:
var_imp = pd.DataFrame(
    {'varname'   : dat_train_x.columns,
    'importance' : list(mod_rf.feature_importances_)})
var_imp.sort_values('importance')#['varname']

## sort variables by importance for plotting:
varname_list = list(var_imp.sort_values('importance')['varname'])
varname_cat = CategoricalDtype(categories = varname_list, ordered=True)
var_imp['varname_cat'] = \
    var_imp['varname'].astype(str).astype(varname_cat)

## plot variable importance (15 most important):
ggplot(var_imp[-15:], aes(y = 'importance', x = 'varname_cat')) + \
    geom_bar(stat = 'identity') + \
    coord_flip()

## ------------------------------------------------------------------------- ##
## partial dependence plots
## ------------------------------------------------------------------------- ##

# Package scikit-learn (PDP via function plot_partial_dependence() ) 
#   http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html
# Package PDPbox (ICE, c-ICE for single and multiple predictors) 
#   https://github.com/SauceCat/PDPbox 

from pdpbox import pdp, get_dataset, info_plots

## [[here]] use merge! (do join!)

#pd.merge(dat_train_x, pd.DataFrame(dat_train_y), left_index = True, right_index = True)
#dat_train_x.join(dat_train_y)  ## identical

## target distribution for numeric feature:
wch_feature = features[1]
fig, axes, summary_df = info_plots.target_plot(
    df = dat_train_x.join(dat_train_y),
    feature = wch_feature,
    feature_name = wch_feature, target = target, 
    show_percentile = True
)

# ## check prediction distribution for numeric feature
# ## (doesn't work?)
# wch_feature = features[1]
# fig, axes, summary_df = info_plots.actual_plot(
#     model = mod_rf, X = dat_train_x, 
#     feature = wch_feature, feature_name = wch_feature, 
#     show_percentile = True
# )

## pdp (and then ice plto) for numeric feature:
wch_feature = features[1]
pdp_current = pdp.pdp_isolate(
    model = mod_rf, dataset = dat_train_x.join(dat_train_y), 
    num_grid_points = 20, n_jobs = -2, ## needs to be 1 for XGBoost model!
    model_features = features, feature = wch_feature
)
fig, axes = pdp.pdp_plot(pdp_current, wch_feature)
## ice-plot for numeric feature:
fig, axes = pdp.pdp_plot(
    pdp_current, wch_feature, plot_lines = True, frac_to_plot = 100,  ## percentate, not fraction! 
    x_quantile = True, plot_pts_dist=True, show_percentile=True
)

## pdp (and then ice plto) for numeric feature:
wch_feature = features[6]
pdp_current = pdp.pdp_isolate(
    model = mod_rf, dataset = dat_train_x.join(dat_train_y), 
    num_grid_points = 20, n_jobs = -2, ## needs to be 1 for XGBoost model!
    model_features = features, feature = wch_feature
)
fig, axes = pdp.pdp_plot(pdp_current, wch_feature)
## ice-plot for numeric feature:
fig, axes = pdp.pdp_plot(
    pdp_current, wch_feature, plot_lines = True, frac_to_plot = 100,  ## percentate, not fraction! 
    x_quantile = True, plot_pts_dist=True, show_percentile=True
)



## [[here]]
## continue with tree inspection

## [[todo]]
## repeat line plots from above but with predictions, in addition!

# ## visualize a single tree:
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = mod_rf.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = dat_x.columns, 
#                 rounded = True, precision = 1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')

## xgboost





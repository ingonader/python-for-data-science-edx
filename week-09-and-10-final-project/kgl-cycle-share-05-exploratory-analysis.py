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

%matplotlib osx

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
                        radius = 2, color = "red")\
    .add_to(folium_map)
## add markers for possible weather stations:
for i in range(0, len(weatherstation_name)):
    folium.Marker(location = weatherstation_latlon[i],
                  popup = weatherstation_name[i])\
    .add_to(folium_map)
## save plot as html:
folium_map.save("map-of-bike-and-possible-weather-stations.html")

## [[todo]]
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


## define window for rolling mean and calculate:
window = 14*24
dat_trip_hr["trip_cnt_rollmean"] = dat_trip_hr[["trip_cnt"]]\
    .rolling(window = window, center = False).\
    mean()

%matplotlib inline
## line plot of number of trips per hour:
p = ggplot(dat_trip_hr, aes(y = 'trip_cnt', x = 'start_date')) + \
    geom_point(alpha = .05) + \
    geom_smooth(method = 'mavg', method_args = {'window' : window}, 
                color = 'red', se = False) + \
    labs(
        title = 'Number of bike trips from 2014 to 2017',
        x = 'Date',
        y = 'Number of trips per hour'
    ) + \
    scale_x_date(date_labels = "%b\n%Y")
    #geom_line(aes(y = 'trip_cnt_rollmean'), color = "lightgreen") + \
print(p)
ggsave(plot = p, filename = os.path.join(path_out, 'expl-trips-per-hour-2014-2017.jpg'), 
       height = 6, width = 6, unit = 'in', dpi = 150)



dat_tmp = dat_trip_hr[["trip_cnt", "trip_cnt_rollmean"]].dropna()

## correlation and explained variance of rolling mean and actual trip count:
rollmean_r = dat_trip_hr[["trip_cnt", "trip_cnt_rollmean"]].corr()
rollmean_r2 = rollmean_r ** 2
rollmean_r2
rollmean_r = dat_tmp.corr()
rollmean_r2 = rollmean_r ** 2
rollmean_r2
r2_score(dat_tmp[["trip_cnt"]], dat_tmp[["trip_cnt_rollmean"]])

## mean absolute error: 
mean_absolute_error(dat_tmp[["trip_cnt"]], dat_tmp[["trip_cnt_rollmean"]])

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

#dat_hr_all.columns
varnames_cor = ['trip_cnt', 'Year', 'Month', 'Temp (°C)', 
                'Dew Point Temp (°C)', 'Rel Hum (%)',
                'Wind Dir (10s deg)', 'Wind Spd (km/h)', 
                'Stn Press (kPa)', 'hr_of_day', 'day_of_week']

dat_cor = dat_hr_all[varnames_cor]

## correlation:
dat_cor.corr()
cormat = dat_cor.corr()

## correlation heatmap:

## simple:
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(cormat, cmap = colormap)

varnames_heatmap_long = [varnames_orig_long_dict[i] for i in cormat.columns]

## more complex:
%matplotlib inline
#%matplotlib osx
fig, ax = plt.subplots(figsize = (10, 10))
#Generate Color Map, red & blue
colormap = sns.diverging_palette(220, 10, as_cmap = True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(cormat, cmap = colormap, annot = True, fmt = ".2f", center = 0)
## Apply axes tickmarks of more explicit variable names:
plt.xticks(np.arange(0, len(cormat.columns)) + .5, varnames_heatmap_long)
plt.yticks(np.arange(0, len(cormat.columns)) + .5, varnames_heatmap_long)
plt.show()

filename_this = "expl-corr-heatmap.jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), 
            dpi = 150, pad_inches = 0.025, bbox_inches = "tight")







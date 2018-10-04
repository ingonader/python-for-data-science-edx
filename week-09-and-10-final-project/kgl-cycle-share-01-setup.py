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




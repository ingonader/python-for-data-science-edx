## ############################################################# ## 
## Analysis of IMDB Movie Dataset
## For EdX Course
## Python for Data Science (Week 6 Mini Project)
## ############################################################# ##

## ============================================================= ##
## import libraries
## ============================================================= ##

import requests
import io
import zipfile
import os
import urllib.parse
import pandas as pd

## ============================================================= ##
## download and extract zip
## ============================================================= ##

## define url and filename, as well as target path:
url_base = 'http://files.grouplens.org/datasets/movielens/'
filename = 'ml-20m.zip'
url = urllib.parse.urljoin(url_base, filename)
path_dat = './data'

## download:
r = requests.get(url, allow_redirects=True)
open(filename, 'wb').write(r.content)

## check download:
os.getcwd()
os.listdir()

## unzip:
zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall(path_dat)
zip_ref.close()

## ============================================================= ##
## Load data files
## ============================================================= ##

dat_movies = pd.read_csv(os.path.join(path_dat, 'ml-20m/movies.csv'), sep = ',')
dat_movies.head(2)

dat_ratings = pd.read_csv(os.path.join(path_dat, 'ml-20m/ratings.csv'), sep = ',')
dat_ratings.head(2)

dat_tags = pd.read_csv(os.path.join(path_dat, 'ml-20m/tags.csv'), sep = ',')
dat_tags.head(2)

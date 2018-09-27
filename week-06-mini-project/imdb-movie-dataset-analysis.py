## ######################################################################### ##
## Analysis of IMDB Movie Dataset
## For EdX Course
## Python for Data Science (Week 6 Mini Project)
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
from plotnine import *
import matplotlib.pyplot as plt


## ========================================================================= ##
## download and extract zip
## ========================================================================= ##

## define url and filename, as well as target path:
url_base = 'http://files.grouplens.org/datasets/movielens/'
filename = 'ml-20m.zip'
url = urllib.parse.urljoin(url_base, filename)
path_dat = './data'

# ## download (commented out in order not to repeat it every time):
# r = requests.get(url, allow_redirects=True)
# open(filename, 'wb').write(r.content)
#
# ## check download:
# os.getcwd()
# os.listdir()
# 
# ## unzip:
# zip_ref = zipfile.ZipFile(filename, 'r')
# zip_ref.extractall(path_dat)
# zip_ref.close()

## ========================================================================= ##
## Load data files
## ========================================================================= ##

dat_movies = pd.read_csv(
    os.path.join(path_dat, 'ml-20m/movies.csv'), 
    sep = ',')

dat_movies.head(2)
dat_movies.info()

#dat_movies.dtype                    ## 'DataFrame' object has no attribute 'dtype'
dat_movies['movieId'].dtype
dat_movies['title'].dtype   

type(dat_movies)                     ## also for DataFrame: pandas.core.frame.DataFrame
type(dat_movies[['movieId']])        ## pandas.core.frame.DataFrame
type(dat_movies['movieId'])          ## pandas.core.series.Series
type(dat_movies['movieId'].values)   ## numpy.ndarray
type(dat_movies['movieId'][1])       ## numpy.int64



dir(dat_movies)                      ## list all methods?

## find strings in this list:
# dir(dat_movies).str.contains('unstack')  ... str.-methods only work on pandas df, not lists
list(filter(lambda x:'values' in x, dir(dat_movies)))  ## filter returns an iterable, hence need 'list'
list(filter(lambda x: re.search(r'unstack', x), dir(dat_movies)))

#cond = df['A'].str.contains('a')

dat_movies.__dict__                  ## lengthy, equivalent to vars(<>)
vars(dat_movies)                     ## lengthy, equivalent to <>.__dict__

## get column names:
list(dat_movies.columns.values)      
list(dat_movies)                     ## same

## summary of data frame
dat_movies.describe()                ## similar to R's summary()

dat_ratings = pd.read_csv(
    os.path.join(path_dat, 'ml-20m/ratings.csv'), 
    sep = ',')

dat_ratings.head(2)
dat_ratings.info()

dat_tags = pd.read_csv(
    os.path.join(path_dat, 'ml-20m/tags.csv'), 
    sep = ',')

dat_tags.head(2)
dat_tags.info()

## Note:
## For some reason, string columns are of type "object".
## https://stackoverflow.com/questions/33957720/how-to-convert-column-with-dtype-as-object-to-string-in-pandas-dataframe
## since strings data types have variable length, 
## it is by default stored as object dtype. If you want to store them as 
## string type, you can do something like this.
## df['column'] = df['column'].astype('|S80') #where the max length is set at 80 bytes,
## or alternatively
## df['column'] = df['column'].astype('|S') # which will by default set the length to the max len it encounters
##  the pandas dataframe stores the pointers to the strings and hence it is of type 'object'.

#dat_movies.head(2)['title'].astype('str')

## ========================================================================= ##
## data prep
## ========================================================================= ##

## possible research questions:
## * what is the relationship of movie complexity 
##   (as measured by number of genres) and average rating? u-shaped?
## * what is the relationship of genres and average rating? 
##   have different genres different ratings, on average?


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## aggregate ratings data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## aggregate ratings data:
## https://stackoverflow.com/questions/38935541/dplyr-summarize-equivalent-in-pandas
dat_ratings_agg = dat_ratings \
    .groupby('movieId') \
    .agg({'rating': ['size', 'min', 'max', 'mean', 'std'], 
         'timestamp': ['min', 'max', 'mean', 'std']})
#dat_ratings_agg.head(2)

## rename columns:
dat_ratings_agg.columns = ['_'.join(col) \
                           for col in dat_ratings_agg.columns]
#dat_ratings_agg.head(2)

## add correct timestamp column (after aggregation, 
## as they cannot be aggregated like numerical values):
dat_tags['parsed_time'] = pd.to_datetime(
    dat_tags['timestamp'], unit='s')
dat_ratings['parsed_time'] = pd.to_datetime(
    dat_ratings['timestamp'], unit='s')
dat_ratings_agg['parsed_time_min'] = pd.to_datetime(
    dat_ratings_agg['timestamp_min'], unit='s')
dat_ratings_agg['parsed_time_max'] = pd.to_datetime(
    dat_ratings_agg['timestamp_max'], unit='s')
dat_ratings_agg['parsed_time_mean'] = pd.to_datetime(
    dat_ratings_agg['timestamp_mean'], unit='s')
dat_ratings_agg.head(2)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## merge data files into one wide file
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## merge data files into one wide file for analysis by movie:
dat_raw = pd.merge(
    left = dat_movies,
    right = dat_ratings_agg,
    how = 'left',
    on = 'movieId')

dat_raw.head(2)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## add measurement for movie complexity
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## add measurement for movie complexity:
dat_raw['complexity'] = dat_raw['genres'] \
    .str.split('|') \
    .apply(lambda x: len(x)) \
    .astype(float)

# dat_raw['complexity']  ## of type object [[!]] when used w/o astype(float)
dat_raw.info()


## exclude movies that have no genres listed:
## '(no genres listed)' --> None
#dat_raw['complexity'] = None if (dat_raw['genres'] == '(no genres listed)') else dat_raw['complexity']
dat_raw['complexity'] = np.where(dat_raw['genres'] == '(no genres listed)', 
                                 None,
                                dat_raw['complexity'])

## turns 'complexity' into type 'object' again...
dat_raw['complexity'] = dat_raw['complexity'].astype(float)

## inspect correctness:
dat_raw.groupby(['genres', 'complexity']).agg({'genres': 'size'})
dat_raw.groupby(['genres', 'complexity']).agg({'genres': 'size'}).sort_values(by = 'genres')
## Note:
## 'None' values are just omitted by groupby?

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## add is_genre attributes for most common genres
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## get list of different genres:
tmp = dat_raw['genres'] \
    .str.split('|')

## similar to unlist, I suppose:
#from itertools import chain
list(chain.from_iterable(tmp))

genres_nonunique = list(chain.from_iterable(tmp))
len(genres_nonunique)

## make frequency table:
# import collections
genre_counter = collections.Counter(genres_nonunique)
print(genre_counter)
print(genre_counter.values())
print(genre_counter.keys())
print(genre_counter.most_common(3))

## make frequency table:
# {x:genres_nonunique.count(x) for x in genres_nonunique}
## (horribly slow, but works)

## create indicator column for each genre:
genre_inds = []
for i in genre_counter.keys():
    #print('creating indicator for key', i, ':')
    this_ind_name = 'is_' + re.sub('[-\(\) ]', '', i).lower()
    genre_inds.append(this_ind_name)
    #print(this_ind_name)
    dat_raw[this_ind_name] = dat_raw['genres'].str.contains(i).astype(int)

## make dictionary of genre_inds: genre_names:
genre_dict = dict(zip(genre_inds, genre_counter.keys()))
genre_dict

#dat_raw.info()
#genre_inds
#dat_raw.head(2)

## ========================================================================= ##
## Data exploration
## ========================================================================= ##

# dat_raw.info()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## generic data exploration
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## [[?]] ask guenther: how to properly use matplotlib in jupyter lab?
dat_raw.hist(bins = 20, size = (10, 10))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## univariate data checks
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## mean ratings:

## check mean ratings (histogram):
ggplot(dat_raw, aes(x = 'rating_mean')) + \
  geom_histogram(bins = 40, color = 'blue', fill = 'blue')

## same plot (histogram) using matplotlib, simple variant:
## (doesn't work with missing values in the data)
## (preliminary) conclusion: does not work with jupyterlab, only with 
## ipython notebooks. No idea why. 
## [[?]] how to get matplotlib plots working, without showing all intermediate steps?

# %matplotlib inline
# %matplotlib ipympl
# %matplotlib widget
plt.hist(dat_raw['rating_mean'].dropna().values, 40, density = False, facecolor = 'blue')
plt.grid(True)
plt.show()

## same plot (histogram) using matplotlib, complex variant:
# fig, ax = plt.subplots()
# plt.hist(dat_raw['rating_mean'], 10, normed=False, facecolor='green')

## complexity:

ggplot(dat_raw, aes(x = 'complexity')) + \
  geom_bar(color = 'blue', fill = 'blue')


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## multivariate checks
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## plot complexity vs. average rating, using ggplot/plotnine:
cor_this = dat_raw[['rating_mean', 'complexity']].corr().iloc[0, 1]
p = ggplot(dat_raw, aes(y = 'rating_mean', x = 'complexity')) + \
  geom_jitter(height = 0.1, width = 0.4, alpha = .1, na_rm = True) + \
  geom_smooth(method = 'glm', color = 'red', na_rm = True) + \
  coord_cartesian(xlim = [1, 10],
                 ylim = [0, dat_raw['rating_mean'].max()]) + \
  scale_x_continuous(breaks = range(1, 11)) + \
  labs(
    title = 'Complexity and mean rating in all movies',
    x = 'Movie complexity (number of genres associated with movie)',
    y = 'Average rating of movie (over all users)'
  ) + \
  annotate(
    geom = 'label', x = 9, y = .5,
    label = 'r = {0:1.2f}'.format(cor_this)
  )
print(p)

ggsave(plot = p, filename = 'plot-scatter-all.jpg', 
       height = 6, width = 6, unit = 'in', dpi = 300)

## similar plot using matplotlib:
## [[todo]]

# ## plot complexity vs. average rating, within genre; using ggplot/plotnine:
genre_inds_plot = list(set(genre_inds) - set(['is_nogenreslisted']))
for i in genre_inds_plot:
    dat_this = dat_raw[dat_raw[i] == True]
    cor_this = dat_this[['rating_mean', 'complexity']].corr().iloc[0, 1]
    p = ggplot(dat_this, aes(y = 'rating_mean', x = 'complexity', )) + \
      geom_jitter(height = 0.1, width = 0.4, alpha = 0.1, na_rm = True) + \
      geom_smooth(method = 'glm', color = 'red', na_rm = True) + \
      coord_cartesian(xlim = [1, 10],
                     ylim = [0, dat_raw['rating_mean'].max()]) + \
      scale_x_continuous(breaks = range(1, 11)) + \
      labs(
        title = 'Complexity and mean rating in genre: {0}'.format(genre_dict[i]),
        x = 'Movie complexity (number of genres associated with movie)',
        y = 'Average rating of movie (over all users)'
      ) + \
      annotate(
        geom = 'label', x = 9, y = .5,
        label = 'r = {0:1.2f}'.format(cor_this)
      )
    print(p)
    ggsave(plot = p, filename = 'plot-scatter-genre-{0}.jpg'.format(i), 
           height = 6, width = 6, unit = 'in', dpi = 300)
    
    
## similar plot using matplotlib:
## [[todo]]

## [[here]][[todo]]
## * correlation in plots do not correspond to correlation in table?!?
##   WHY?!?


## ========================================================================= ##
## Analysis
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## correlation
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

dat_nona = dat_raw.dropna()

## np.corrcoef(dat_nona['rating_mean'], dat_nona['complexity'])
## AttributeError: 'float' object has no attribute 'shape'
## The error is reproducible if the array is of dtype=object

## correlation over all movies (using numpy):
np.corrcoef(dat_nona['rating_mean'], dat_nona['complexity'].astype(float))

## correlation over all movies (using pandas):
## (can handle missings, somehow)
dat_nona[['rating_mean', 'complexity']].corr()
dat_raw[['rating_mean', 'complexity']].corr()

## correlation within each movie category:
dat_cor = pd.DataFrame([])
for i in genre_inds:
    dat_this = dat_raw[dat_raw[i] == True]
    cor_this = dat_this[['rating_mean', 'complexity']].corr().iloc[0, 1]
    dat_cor = dat_cor.append(pd.DataFrame(
        {'Variable': i,
         'Genre'   : genre_dict[i],
         'n'       : dat_this.shape[0],
         'r'       : cor_this.round(3)}, 
        index = [0]))
    
dat_cor.sort_values(by = 'r', ascending = False)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## regression
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## mostly from 
## http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy ## for design matrices like R

## ------------------------------------------------------------------------- ##
## define features and formula
## ------------------------------------------------------------------------- ##

## define target and features:
target = 'rating_mean'
features = [
 'complexity',
 'is_adventure',
 'is_animation',
 'is_children',
 'is_comedy',
 'is_fantasy',
 'is_romance',
 'is_drama',
 'is_action',
 'is_crime',
 'is_thriller',
 'is_horror',
 'is_mystery',
 'is_scifi',
 'is_imax',
 'is_documentary',
 'is_war',
 'is_musical',
 'is_western',
 'is_filmnoir'
 ## 'is_nogenreslisted'  ## excluded, since complexity is always 1 here
]
# list(dat_raw)

dat_raw.info()

# ## formula as text for patsy: without interactions
# formula_txt = target + ' ~ ' + ' + '.join(features)

## formula as text for patsy: with interactions
formula_txt = target + ' ~ ' + \
    ' + '.join(features) + ' + ' + \
    ' + complexity:'.join(list(set(features) - set(['complexity'])))
formula_txt

## create design matrices using patsy (could directly be used for modeling):
#patsy.dmatrix?
dat_y, dat_x = patsy.dmatrices(formula_txt, dat_raw, 
                               NA_action = 'drop',
                               return_type = 'dataframe')
#dat_x.design_info
#dat_x

## ------------------------------------------------------------------------- ##
## train / test split
## ------------------------------------------------------------------------- ##

# ## Split the data into training/testing sets:
# dat_train_x, dat_test_x, dat_train_y, dat_test_y = train_test_split(
#    dat_nona[features], dat_nona[target], test_size = 0.33, random_state = 142)

# ## shapes w/o using patsy/dmatrices:
# dat_train_x.shape  # (15231, 21)
# dat_test_x.shape   # (7503, 21)
# dat_train_y.shape  # (15231,)      # type: pandas.core.series.Series

## Split the data into training/testing sets (using patsy/dmatrices):
dat_train_x, dat_test_x, dat_train_y, dat_test_y = train_test_split(
    dat_x, dat_y, test_size = 0.33, random_state = 142)

# ## shapes wwith using patsy/dmatrices:
# dat_train_x.shape  # (17756, 41)
# dat_test_x.shape   # (8746, 41)
# dat_train_y.shape  # (17756, 1)    # type: pandas.core.frame.DataFrame

## convert y's to Series (to match data types between patsy and non-patsy data prep:)
dat_train_y = dat_train_y[target]
dat_test_y = dat_test_y[target]

## [[?]] [[todo]]
## normalize input for scikit-learn regression?

## ------------------------------------------------------------------------- ##
## normalize data
## ------------------------------------------------------------------------- ##

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##
## by hand
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##

# ## variables to normalize (e.g., ignore intercept):
# varnames_normalize_x = list(set(dat_train_x.columns) - set(['Intercept']))
# 
# ## get mean and sd from training data:
# ztrans_mean_x = dat_train_x[varnames_normalize_x].apply(np.mean)
# ztrans_sd_x = dat_train_x[varnames_normalize_x].apply(np.std)
# ztrans_mean_y = np.mean(dat_train_y)
# ztrans_sd_y = np.std(dat_train_y)
# 
# ## replace sd's that are NaN with 1 (just don't transform):
# ## [[todo]]
# 
# ## z-standardize data (train):
# dat_train_x[varnames_normalize_x] = (dat_train_x[varnames_normalize_x] - ztrans_mean_x) / ztrans_sd_x
# dat_train_y = (dat_train_y - ztrans_mean_y) / ztrans_sd_y
# 
# ## z-standardize data (test):
# dat_test_x[varnames_normalize_x] = (dat_test_x[varnames_normalize_x] - ztrans_mean_x) / ztrans_sd_x
# dat_test_y = (dat_test_y - ztrans_mean_y) / ztrans_sd_y

## produces warning:
## /Users/ingonader/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3137: SettingWithCopyWarning: 
## A value is trying to be set on a copy of a slice from a DataFrame.
## Try using .loc[row_indexer,col_indexer] = value instead
## 
## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable## /indexing.html#indexing-view-versus-copy
##   self[k1] = value[k2]

## [[?]] ask what to do as best practice!


## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##
## using StandardScaler
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##

# ## variables to normalize (e.g., ignore intercept):
# #varnames_normalize_x = list(set(dat_train_x.columns) - set(['Intercept']))
# 
# ## Standardize features by removing the mean and scaling to unit variance:
# scaler = StandardScaler()
# scaler.fit(dat_train_x)
# #scaler.fit(dat_train_x[varnames_normalize_x])
# 
# print(scaler)
# vars(scaler)
# 
# print(scaler.mean_)
# print(scaler.var_)
# 
# dat_train_x = scaler.transform(dat_train_x)
# type(dat_train_x)                  ## numpy.ndarray
# pd.DataFrame(dat_train_x).head()   ## varnames are lost... 

## [[?]] ask guenther: is there a good way to scale variables with pre-built function?

## ------------------------------------------------------------------------- ##
## estimate model and evaluate fit and model assumptions
## ------------------------------------------------------------------------- ##

## Create linear regression object
mod_01 = linear_model.LinearRegression()

## Train the model using the training sets
mod_01.fit(dat_train_x, dat_train_y)

## Make predictions using the testing set
dat_test_pred = mod_01.predict(dat_test_x)
dat_train_pred = mod_01.predict(dat_train_x)

## Inspect model:
mean_squared_error(dat_train_y, dat_train_pred)  # MSE in training set
mean_squared_error(dat_test_y, dat_test_pred)    # MSE in test set
r2_score(dat_train_y, dat_train_pred)            # R^2 (r squared) in test set
r2_score(dat_test_y, dat_test_pred)              # R^2 (r squared) in test set

## VIF:
## For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [
    variance_inflation_factor(dat_train_x.values, i) \
    for i in range(dat_train_x.shape[1])]
vif["features"] = dat_train_x.columns
vif

## vif is Inf when complexity is in the model. This is despite
## the fact that the correlations aren't too high.
## [[todo]] should try with R for comparison.

## correlation between features:
cormat = dat_train_x.corr()
cormat.round(2)

## how many correlations bigger than...:
np.sum(cormat > .90)
cormat.loc[cormat.loc['is_imax'] > .90, 'is_imax']
cormat.loc[cormat.loc['is_adventure'] > .90, 'is_adventure']

## inspect model coefficients:
## [[?]] something's wrong here
mod_01.coef_                                # coefficients
mod_01.coef_[0]
coefs = pd.DataFrame({
    'coef' : dat_train_x.columns,
    'value': mod_01.coef_[0]       ## without the [0] when not using patsy
})
coefs


## calculate residuals:
dat_train_resid = dat_train_y - mod_01.predict(dat_train_x)
dat_train_resid.describe()

## fortify training data (when using target variables of type Series):
dat_train_fortify = pd.DataFrame({
    'y':     dat_train_y,                 # Length: 15231, dtype: float64
    'pred' : mod_01.predict(dat_train_x), # array([3.27736039, 3.27302689, 3.27302689, ..., ])
    'resid': dat_train_resid              # Length: 15231, dtype: float64
})
type(dat_train_y)                  # pandas.core.series.Series
type(mod_01.predict(dat_train_x))  # numpy.ndarray
type(dat_train_resid)              # pandas.core.series.Series

## normality of residuals (training data):
ggplot(
    dat_train_fortify, aes(x = 'resid')) + \
    geom_histogram(bins = 40, fill = 'blue')

## plot residuals vs. predicted values (training data):
ggplot(
    dat_train_fortify, aes(x = 'pred', y = 'resid')) + \
    geom_point(alpha = .1) + \
    geom_smooth(color = 'blue')

# ## Residual Plot using yellowbrick (doesn't quite work):
# ## http://www.scikit-yb.org/en/latest/api/regressor/residuals.html
# from yellowbrick.regressor import ResidualsPlot
# visualizer = ResidualsPlot(mod_01)
# visualizer.fit(dat_train_x, dat_train_y)
# visualizer.score(dat_test_x, dat_test_y)
# visualizer.poof()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## linear regression using statsmodels
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

import statsmodels.api as sm

mod_linreg_sm = sm.OLS(dat_train_y, dat_train_x)
mod_linreg_sm = mod_linreg_sm.fit()

mod_linreg_sm.summary()

## a lot of stuff can be extracted:
dir(mod_linreg_sm)

print('Parameters: ', mod_linreg_sm.params)
print('R2: ', mod_linreg_sm.rsquared)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ridge regression
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures

# ## add interaction terms:
# ## (not as cool; better use patsy)
# poly = PolynomialFeatures(interaction_only = True,include_bias = False)
# poly.fit_transform(dat_train_x).shape

## fit model:
mod_ridge = Ridge(alpha=1.0)
mod_ridge.fit(dat_train_x, dat_train_y)

## [[?]] What about intercept? Is that regularized? (shouldn't be)

## Make predictions using the testing set
dat_test_pred = mod_ridge.predict(dat_test_x)
dat_train_pred = mod_ridge.predict(dat_train_x)

## Inspect model:
mean_squared_error(dat_train_y, dat_train_pred)  # MSE in training set
mean_squared_error(dat_test_y, dat_test_pred)    # MSE in test set
r2_score(dat_train_y, dat_train_pred)            # R^2 (r squared) in test set
r2_score(dat_test_y, dat_test_pred)              # R^2 (r squared) in test set

## inspect coefficients:
mod_ridge.coef_                                # coefficients
mod_ridge.coef_[0]
coefs = pd.DataFrame({
    'coef' : dat_train_x.columns,
    'value': mod_ridge.coef_ 
})
coefs

## plot coefficients:
ggplot(coefs, aes(x = 'coef', y = 'value')) + geom_bar(stat = 'identity') + coord_flip()

## calculate residuals:
dat_train_resid = dat_train_y - mod_ridge.predict(dat_train_x)
dat_train_resid.describe()

## fortify training data:
dat_train_fortify = pd.DataFrame({
    'y':     dat_train_y,
    'pred' : mod_ridge.predict(dat_train_x),
    'resid': dat_train_resid
})

## normality of residuals (training data):
ggplot(
    dat_train_fortify, aes(x = 'resid')) + \
    geom_histogram(bins = 40, fill = 'blue')

## plot residuals vs. predicted values (training data):
ggplot(
    dat_train_fortify, aes(x = 'pred', y = 'resid')) + \
    geom_point(alpha = .1) + \
    geom_smooth(color = 'blue')


## [[todo]]
## * normalize data for both linear and ridge regression
## * add matplotlib plots
## * prettify plots for presentation
## * make presentation


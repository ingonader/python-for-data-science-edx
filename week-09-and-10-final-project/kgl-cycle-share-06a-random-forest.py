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
## modeling number of trips
## ========================================================================= ##

## using all data (as opposed to using only data with only non-zero trips):

import patsy ## for design matrices like R
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import pathvalidate as pv

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

# ## try all twofold interactions, in order to 
# ## find important ones via variable importance plots:
# formula_txt = target + ' ~ (' + \
#     ' + '.join(features) + ') ** 2 - 1'
# formula_txt

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
p = ggplot(var_imp[-15:], aes(y = 'importance', x = 'varname_cat')) + \
    geom_bar(stat = 'identity') + \
    coord_flip()
print(p)

filename_this = 'plot-variable-importance.jpg'
#filename_this = 'plot-variable-importance-with-interactions.jpg'
ggsave(plot = p, 
       filename = os.path.join(path_out, filename_this),
       height = 6, width = 6, unit = 'in', dpi = 300)

## ------------------------------------------------------------------------- ##
## partial dependence plots: main effects
## ------------------------------------------------------------------------- ##

from pdpbox import pdp, get_dataset, info_plots

# Package scikit-learn (PDP via function plot_partial_dependence() ) 
#   http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html
# Package PDPbox (ICE, c-ICE for single and multiple predictors) 
#   https://github.com/SauceCat/PDPbox 

#pd.merge(dat_train_x, pd.DataFrame(dat_train_y), left_index = True, right_index = True)
#dat_train_x.join(dat_train_y)  ## identical

## pdp (and then ice plot) calculation for numeric feature:
#features[1]
wch_feature = "Q('Temp (°C)')"
pdp_current = pdp.pdp_isolate(
    model = mod_rf, dataset = dat_train_x.join(dat_train_y), 
    num_grid_points = 20, n_jobs = -2, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, 
    feature = wch_feature
)

## pdp plot for numeric features:
fig, axes = pdp.pdp_plot(pdp_current, wch_feature)
filename_this = "pdp-main---" + pv.sanitize_python_var_name(wch_feature) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), dpi = 300)

## ice-plot for numeric feature:
fig, axes = pdp.pdp_plot(
    pdp_current, wch_feature, plot_lines = True, frac_to_plot = 100,  ## percentate, not fraction! 
    x_quantile = True, plot_pts_dist=True, show_percentile=True)
filename_this = "ice-main---" + pv.sanitize_python_var_name(wch_feature) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), dpi = 300)

## [[here]] [[?]] how to set axis labels?



## pdp (and then ice plot) calculation for numeric feature:
features[6]
wch_feature = "Q('hr_of_day')"
pdp_current = pdp.pdp_isolate(
    model = mod_rf, dataset = dat_train_x.join(dat_train_y), 
    num_grid_points = 20, n_jobs = -2, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, 
    feature = wch_feature
)

## pdp plot for numeric features:
fig, axes = pdp.pdp_plot(pdp_current, wch_feature)
filename_this = "pdp-main---" + pv.sanitize_python_var_name(wch_feature) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), dpi = 300)

## ice-plot for numeric feature:
fig, axes = pdp.pdp_plot(
    pdp_current, wch_feature, plot_lines = True, frac_to_plot = 100,  ## percentate, not fraction! 
    x_quantile = True, plot_pts_dist=True, show_percentile=True)
filename_this = "ice-main---" + pv.sanitize_python_var_name(wch_feature) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), dpi = 300)


## ------------------------------------------------------------------------- ##
## partial dependence plots: interactions
## ------------------------------------------------------------------------- ##

#[features[6], features[5]]
wch_features = ["Q('hr_of_day')", "Q('Stn Press (kPa)')"]
inter_current = pdp.pdp_interact(
    model = mod_rf, dataset = dat_train_x.join(dat_train_y),
    num_grid_points = [10, 10], n_jobs = -2, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, features = wch_features)
fig, axes = pdp.pdp_interact_plot(
    inter_current, wch_features, x_quantile = True, 
    plot_type = 'contour', plot_pdp = False
)
filename_this = "pdp-interact---" + \
    pv.sanitize_python_var_name(wch_features[0]) + "--" + \
    pv.sanitize_python_var_name(wch_features[1]) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), dpi = 300)

#[features[6], features[7]]
wch_features = ["Q('hr_of_day')", "Q('day_of_week')"]
inter_current = pdp.pdp_interact(
    model = mod_rf, dataset = dat_train_x.join(dat_train_y),
    num_grid_points = [10, 10], n_jobs = -2, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, features = wch_features)
fig, axes = pdp.pdp_interact_plot(
    inter_current, wch_features, x_quantile = True, 
    plot_type = 'contour', plot_pdp = False
)
filename_this = "pdp-interact---" + \
    pv.sanitize_python_var_name(wch_features[0]) + "--" + \
    pv.sanitize_python_var_name(wch_features[1]) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), dpi = 300)

#[features[1], features[2]]
wch_features = ["Q('Temp (°C)')", "Q('Rel Hum (%)')"]
inter_current = pdp.pdp_interact(
    model = mod_rf, dataset = dat_train_x.join(dat_train_y),
    num_grid_points = [10, 10], n_jobs = -2, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, features = wch_features)
fig, axes = pdp.pdp_interact_plot(
    inter_current, wch_features, x_quantile = True, 
    plot_type = 'contour', plot_pdp = False
)
filename_this = "pdp-interact---" + \
    pv.sanitize_python_var_name(wch_features[0]) + "--" + \
    pv.sanitize_python_var_name(wch_features[1]) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_this), dpi = 300)

## ------------------------------------------------------------------------- ##
## other plots
## ------------------------------------------------------------------------- ##


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




## [[todo]] 
## * save matplotlib plot! how? [[?]]
## * repeat line plots from above but with predictions, in addition!
## * xgboost
## * some categorical prediction model, in order to try out stuff like f1, confusionmatrix, roc curve

## [[here]]
## continue with tree inspection





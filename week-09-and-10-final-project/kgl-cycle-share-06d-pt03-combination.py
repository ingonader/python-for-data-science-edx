## ######################################################################### ##
## Analysis of 
## For EdX Course
## Python for Data Science (Week 9 and 10 Final Project)
## ######################################################################### ##

## Classification first, then regression: combine the two!

## ========================================================================= ## 
## import libraries
## ========================================================================= ##

import patsy ## for design matrices like R
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score  ## area under prec-rec-curve?
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from scipy import stats

import pathvalidate as pv

import joblib

## ========================================================================= ##
## load the two models
## ========================================================================= ##

filename_model_01 = 'model_class_gb.pkl'
filename_model_02 = 'model_nonzero_gradient_boosting.pkl'
filename_out_prefix = 'mod_gbc_gbr_'
n_jobs = 1


## load models:
mod_this_01 = joblib.load(os.path.join(path_out, filename_model_01))
mod_this_02 = joblib.load(os.path.join(path_out, filename_model_02))

## ========================================================================= ##
## make predictions and get model performance
## ========================================================================= ##

## Make predictions for the test set using model 1:
## (probabilities with a certain threshold):
wch_class = 1
dat_train_01_predprob = mod_this_01.predict_proba(dat_train_x)[:, wch_class]  ## numpy array (!)
dat_test_01_predprob =  mod_this_01.predict_proba(dat_test_x)[:,wch_class]    ## numpy array (!)

## convert to classification
## (0/1 only, does not incorporate labels yet):
cutoff = .99
dat_train_pred_01 = (dat_train_01_predprob >= cutoff).astype(int)  
dat_test_pred_01 =  (dat_test_01_predprob  >= cutoff).astype(int)

#np.sum(dat_test_pred_01 == 0)

## Make predictions for the test set using model 2:
dat_test_pred_02 =  mod_this_02.predict(dat_test_x)
dat_train_pred_02 = mod_this_02.predict(dat_train_x)


## for those where predicitons are 1 (having bike rides this hour),
## use the predictions of the regression model
## (otherwise, use zero):
dat_test_pred = np.where(dat_test_pred_01 == 0, ## if classification says "No rides"...
                        0,                      ## 0, otherwise...
                        dat_test_pred_02)       ## take prediction from nonzero model
dat_train_pred = np.where(dat_train_pred_01 == 0, ## if classification says "No rides"...
                          0,                      ## 0, otherwise...
                          dat_train_pred_02)      ## take prediction from nonzero model


## Inspect model:
mean_squared_error(dat_train_y, dat_train_pred)  # MSE in training set
mean_squared_error(dat_test_y, dat_test_pred)    # MSE in test set
mean_absolute_error(dat_train_y, dat_train_pred) # MAE in training set
mean_absolute_error(dat_test_y, dat_test_pred)   # MAE in test set
r2_score(dat_train_y, dat_train_pred)            # R^2 (r squared) in test set
r2_score(dat_test_y, dat_test_pred)              # R^2 (r squared) in test set

## doesn't doo much good...
## only a few classifications predict "no ride", 
## hence, not much use.



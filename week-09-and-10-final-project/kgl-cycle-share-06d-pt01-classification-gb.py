## ######################################################################### ##
## Analysis of 
## For EdX Course
## Python for Data Science (Week 9 and 10 Final Project)
## ######################################################################### ##

## Classification first, then regression

## ========================================================================= ## 
## import libraries
## ========================================================================= ##


import patsy ## for design matrices like R
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import train_test_split

import pathvalidate as pv

## ========================================================================= ##
## modeling trips > 0 and then number of trips
## ========================================================================= ##

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

## define target and features:
target = 'trip_ind'
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
# formula_txt = target + ' ~ (' + ' + '.join(features) + ') ** 2 - 1'
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
mod_class = GradientBoostingClassifier(n_estimators = 100, 
                                   random_state = 42,
                                   loss = 'deviance',
                                   learning_rate = 0.1,
                                   max_depth = 10, 
                                   min_samples_split = 70,
                                   min_samples_leaf = 30,
                                   verbose = 1)

## Train the model using the training sets:
mod_class.fit(dat_train_x, dat_train_y)

## [[?]] missing: how to plot oob error by number of trees, like in R?

## Make predictions of probability:
wch_class = 1
dat_train_predprob = mod_class.predict_proba(dat_train_x)[:, wch_class]  ## numpy array (!)
dat_test_predprob = mod_class.predict_proba(dat_test_x)[:, wch_class]    ## numpy array (!)

## Make predictions using the testing set
dat_train_pred = mod_class.predict(dat_train_x) ## produces 0/1 numpy array (!)
dat_test_pred = mod_class.predict(dat_test_x)   ## produces 0/1 numpy array (!)

## [[todo]]
## precision by cutoff plots


## Inspect model:

## area under the curve:
roc_auc_score(y_true = dat_train_y, y_score = dat_train_predprob)
roc_auc_score(y_true =  dat_test_y, y_score =  dat_test_predprob)

## roc curve:
fpr, tpr, threshold = roc_curve(dat_train_y, dat_train_predprob)
dat_roc_train = pd.DataFrame({"fpr" : fpr,
                       "tpr" : tpr,
                       "threshold" : threshold,
                       "split" : "train"})
fpr, tpr, threshold = roc_curve(dat_test_y, dat_test_predprob)
dat_roc_test = pd.DataFrame({"fpr" : fpr,
                       "tpr" : tpr,
                       "threshold" : threshold,
                       "split" : "test"})
dat_roc = pd.concat([dat_roc_train, dat_roc_test], axis = 0)
ggplot(dat_roc, aes(y = "tpr", x = "fpr", color = "split")) + \
    geom_abline(slope = 1, intercept = 0, color = "grey", linetype = "dashed") + \
    geom_line()

## log-loss (cross-entropy):
log_loss(y_true = dat_train_y, y_pred = dat_train_predprob)
log_loss(y_true =  dat_test_y, y_pred =  dat_test_predprob)


## confusion matrix:
confusion_matrix(y_true = dat_train_y, y_pred = dat_train_pred) ## works fine with numpy arrays and series
confusion_matrix(y_true = dat_test_y,  y_pred = dat_test_pred)
             
## [[?]] https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
## confusion matrix from pandas_ml, more beautiful:
ConfusionMatrix(y_true = dat_train_y.values, y_pred = dat_train_pred)
ConfusionMatrix(y_true = dat_test_y.values,  y_pred = dat_test_pred)
## [[!]] Note: only works with numpy arrays!
## The following won't work:
# ConfusionMatrix(y_true = dat_test_y,  y_pred = dat_test_pred)
# ConfusionMatrix(y_true = dat_test_y,  y_pred = pd.Series(dat_test_pred))

## default classifcation accuracy:
mod_class.score(dat_train_x, dat_train_y)
mod_class.score(dat_test_x,  dat_test_y)
accuracy_score(dat_train_y, dat_train_pred)
accuracy_score(dat_test_y,  dat_test_pred)

## precision:
precision_score(dat_train_y, dat_train_pred)           
precision_score(dat_test_y,  dat_test_pred)             

## recall:
recall_score(dat_train_y, dat_train_pred)           
recall_score(dat_test_y,  dat_test_pred)             

## f1:
f1_score(dat_train_y, dat_train_pred)           
f1_score(dat_test_y,  dat_test_pred)             

## ------------------------------------------------------------------------- ##
## save model to disk
## ------------------------------------------------------------------------- ##

## [[?]] who to persist models?
## * don't use pickle or joblib (unsafe and not persistent)
##   see https://pyvideo.org/pycon-us-2014/pickles-are-for-delis-not-software.html or
##   http://scikit-learn.org/stable/modules/model_persistence.html
##   (3.4.2. Security & maintainability limitations)

from sklearn.externals import joblib

# model_name = 
filename_model = 'model_gradient_boosting.pkl'
joblib.dump(mod_gb, os.path.join(path_out, filename_model))

# ## load:
# filename_model = 'model_gradient_boosting.pkl'
# mod_this = joblib.load(os.path.join(path_out, filename_model))


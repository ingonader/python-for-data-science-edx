## ######################################################################### ##
## Redo scikit-learn ML models from EdX-course 
## with R's mlr package
## ######################################################################### ##

# parallelStop(); rm(list = ls(), inherits = TRUE); rstudioapi::restartSession()

## [[todo]]
## * on.learner.error option
## * parameter tuning in outer tuning loop + inner loop

## ========================================================================= ##
## load packages 
## ========================================================================= ##

library(magrittr)
library(tidyverse)
library(ggplot2)
library(mlr)
library(feather)

## ========================================================================= ##
## global variables and options
## ========================================================================= ##

path_dat <- "/Users/ingonader/data-um-sync/training/coursera-work/python-for-data-science-edx/week-09-and-10-final-project/data"
filename <- "dat_hr_all.feather"

options(tibble.width = Inf)


## ========================================================================= ##
## read data (preprocessed with python)
## ========================================================================= ##

## data preprocessing scripts:
## 
## kgl-cycle-share-01-setup.py
## kgl-cycle-share-02-data-download.py
## kgl-cycle-share-03-data-load.py
## kgl-cycle-share-04-data-prep.py

dat_hr_all <- read_feather(file.path(path_dat, filename))
head(dat_hr_all)
dim(dat_hr_all)

## ========================================================================= ##
## data prep
## ========================================================================= ##

## adapt variable names for R / mlr:
## (otherwise, creating a task will throw an error)
names(dat_hr_all) <- make.names(names(dat_hr_all)) %>% 
  tolower() %>%
  stringr::str_replace_all("\\.{2,}.*", "") %>% ## replace units with ""
  stringr::str_replace_all("\\.", "_")  ## replace dots with underscore


## ========================================================================= ##
## define variables and data subsets (including missing)
## ========================================================================= ##

varnames_target <- 'trip_cnt'
varnames_features <- c('month',
                       'temp',
                       # 'Dew Point Temp (°C)', ## -- exclude, because highly correlated with Temp
                       'rel_hum',
                       'wind_dir',
                       'wind_spd',
                       'stn_press',
                       'hr_of_day',
                       'day_of_week')
varnames_model <- union(varnames_target, varnames_features)

## missing values:
## [[todo]] -- expand here?
dat_hr_mod <- na.omit(dat_hr_all[varnames_model])
dim(dat_hr_mod)

## train and test set (indices):
idx_train <- sample(1:nrow(dat_hr_mod), size = 1000, replace = FALSE) ## size = 26168 ## 10000 for testing
idx_test <- setdiff(1:nrow(dat_hr_mod), idx_train)
length(idx_train)


## ========================================================================= ##
## restore everything from disk
## ========================================================================= ##

## don't save any objects until here:
obj_notsave <- ls()

# load(file = file.path(path_dat, "kgl-mlr-trials_v001.Rdata"))

## ========================================================================= ##
## define task
## ========================================================================= ##


## create a task: (= data + meta-information)
task_full <- makeRegrTask(id = "trip_cnt_mod", 
                     data = dat_hr_mod[varnames_model],
                     target = varnames_target)
task <- subsetTask(task = task_full, subset = idx_train)

## ========================================================================= ##
## create a single learner with random parameter search and CV
## ========================================================================= ##

## enable parallel execution
library(parallelMap)
parallelGetRegisteredLevels()
parallelStartMulticore(cpus = 3, level = "mlr.resample")

## set random seed, also valid for parallel execution:
set.seed(4271, "L'Ecuyer")

## choose resampling strategy:
rdesc <- makeResampleDesc(predict = "both", 
                          method = "CV", iters = 3)
#method = "RepCV", reps = 3, folds = 5)

## parameters for parameter tuning:
ctrl <- makeTuneControlRandom(maxit = 40)
tune_measures <- list(rmse, mae, rsq, timetrain, timepredict)

## standard random forest implementation:
tune_results_rf <- tuneParams(
  "regr.randomForest", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
    makeIntegerParam("nodesize", lower = 10, upper = 50),
    makeIntegerParam("ntree", lower = 100, upper = 500)
  )
)
tune_results_rf
tune_results_rf$x

## faster random forest implementation:
tune_results_ranger <- tuneParams(
  "regr.ranger", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
    makeIntegerParam("min.node.size", lower = 10, upper = 50),
    makeIntegerParam("num.trees", lower = 100, upper = 500)
  )
)
tune_results_ranger

## gradient boosting
tune_results_gbm <- tuneParams(
  "regr.gbm", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("interaction.depth", lower = 1, upper = 9),
    makeIntegerParam("n.minobsinnode", lower = 10, upper = 50),
    makeIntegerParam("n.trees", lower = 100, upper = 1000)
  )
)
tune_results_gbm
#getParamSet("regr.gbm")

## gradient boosting using xgboost:
tune_results_xgboost <- tuneParams(
  "regr.xgboost", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("max_depth", lower = 1, upper = 9),
    makeIntegerParam("nrounds", lower = 100, upper = 1000)
    
  )
)
tune_results_xgboost
#getParamSet("regr.xgboost")

parallelStop()

## ========================================================================= ##
## refit all learners with their tuned parameters (with CV)
## ========================================================================= ##

## start parallelization on benchmark level:
#parallelStartMulticore(cpus = 3, level = "mlr.benchmark")
parallelStartMulticore(cpus = 3, level = "mlr.resample")

## set random seed, also valid for parallel execution:
set.seed(4271, "L'Ecuyer")

rdesc_bm <- makeResampleDesc(predict = "both", 
                             method = "RepCV", reps = 3, folds = 4)

lrns_tuned <- list(
  makeLearner("regr.randomForest",  par.vals = tune_results_rf$x),
  makeLearner("regr.ranger", par.vals = tune_results_ranger$x),
  makeLearner("regr.gbm", par.vals = tune_results_gbm$x),
  makeLearner("regr.xgboost", par.vals = tune_results_xgboost$x)
)

## create training aggregation measures:
rmse.train.mean <- setAggregation(rmse, train.mean)
mae.train.mean <- setAggregation(mae, train.mean)
rsq.train.mean <- setAggregation(rsq, train.mean)

## refit models on complete training data, validate on test data:
bmr_train <- benchmark(
  lrns_tuned, task, rdesc_bm,
  measures = list(rmse, #rmse.train.mean,
                  mae, #mae.train.mean,
                  rsq, #rsq.train.mean,
                  timetrain, timepredict)
)
bmr_train


## ========================================================================= ##
## use tuning wrappers in benchmark itself
## ========================================================================= ##

## also makes it possible to combine with different preprocessing

## [[here]] 

## set random seed, also valid for parallel execution:
set.seed(4271, "L'Ecuyer")

## parameters for parameter tuning (search strategy and iterations):
ctrl <- makeTuneControlRandom(maxit = 40)
tune_measures <- list(rmse, mae, rsq, timetrain, timepredict)

## choose resampling strategy for parameter tuning:
rdesc_tune <- makeResampleDesc(predict = "both", 
                               method = "CV", iters = 3)
#method = "RepCV", reps = 3, folds = 5)

## make tuning wrapper for learner, 
## so it will be tuned during the benchmark later:

## tuner wrapper for standard random forest:
tuner_rf <- mlr::makeTuneWrapper(
  learner = "regr.randomForest",
  resampling = rdesc_tune, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
    makeIntegerParam("nodesize", lower = 10, upper = 50),
    makeIntegerParam("ntree", lower = 100, upper = 500)
  )
)

## tuner wrapper for faster random forest implementation:
tuner_ranger <- mlr::makeTuneWrapper(
  learner = "regr.ranger",
  resampling = rdesc_tune, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
    makeIntegerParam("min.node.size", lower = 10, upper = 50),
    makeIntegerParam("num.trees", lower = 100, upper = 500)
  )
)

## tuner wrapper for gradient boosting:
tuner_gbm <- mlr::makeTuneWrapper(
  learner = "regr.gbm",
  resampling = rdesc_tune, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("interaction.depth", lower = 1, upper = 9),
    makeIntegerParam("n.minobsinnode", lower = 10, upper = 50),
    makeIntegerParam("n.trees", lower = 100, upper = 1000)
  )
)

## tuner wrapper for gradient boosting using xgboost:
tuner_xgboost <- mlr::makeTuneWrapper(
  learner = "regr.xgboost",
  resampling = rdesc_tune, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("max_depth", lower = 1, upper = 9),
    makeIntegerParam("nrounds", lower = 100, upper = 1000)
    
  )
)


## make a list of tuner wrappers:
lrns_tunewrap <- list(
  tuner_rf,
  tuner_ranger,
  tuner_gbm,
  tuner_xgboost
)
## warning produced by:
#mlr::makeLearner("regr.xgboost")

rdesc_bm <- mlr::makeResampleDesc(predict = "both", 
                               method = "RepCV", reps = 3, folds = 4)

#mlr::listMeasures(task)
bmr_tunewrap <- benchmark(
  lrns_tunewrap, task, rdesc_bm,
  # measures = list(rmse, mae, rsq)
  measures = list(rmse, #rmse.train.mean,
                  mae, #mae.train.mean,
                  rsq, #rsq.train.mean)
                  timetrain, timepredict)
)
bmr_tunewrap


## save everything except data and path
obj <- setdiff(ls(), obj_notsave)
#save(obj, file = file.path(path_dat, "kgl-mlr-trials_v001.Rdata"))

## ========================================================================= ##
## visualizing benchmark results
## ========================================================================= ##

plotBMRBoxplots(bmr_train, measure = mae, style = "violin") +
  aes(fill = learner.id) + geom_point(alpha = .5)

plotBMRBoxplots(bmr_tunewrap, measure = mae, style = "violin") +
  aes(fill = learner.id) + geom_point(alpha = .5)


## ========================================================================= ##
## refit all tuned learners on the whole training set
## ========================================================================= ##

## and estimate performance on an identical test set:
rdesc_bmf <- makeFixedHoldoutInstance(train.inds = idx_train,
                                test.inds = idx_test,
                                size = length(c(idx_train, idx_test)))
rdesc_bmf

## refit models on complete training data, validate on test data:
bmr_full <- benchmark(
  lrns_tuned, task_full, rdesc_bmf,
  # measures = list(rmse, mae, rsq)
  measures = list(rmse, #rmse.train.mean,
                  mae, #mae.train.mean,
                  rsq, #rsq.train.mean,
                  timetrain, timepredict)
)
bmr_full

parallelStop()




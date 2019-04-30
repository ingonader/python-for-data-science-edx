## ######################################################################### ##
## Redo scikit-learn ML models from EdX-course 
## with R's mlr package
## ######################################################################### ##

# parallelMap::parallelStop()
# rm(list = ls(), inherits = TRUE); rstudioapi::restartSession()

## [[todo]]
## * on.learner.error option
## * use 4-fold CV in parameter tuning for comparison reasons (with python code)
## * save ggplot plots as files
## * create presentation
## * inspect model using pdp package (and maye ICEbox) (if needed)



## ========================================================================= ##
## load packages 
## ========================================================================= ##

library(magrittr)
library(tidyverse)
library(ggplot2)
library(mlr)
library(feather)
library(tictoc)

## ========================================================================= ##
## global variables and options
## ========================================================================= ##

path_raw <- "/Users/ingonader/data-um-sync/training/coursera-work/python-for-data-science-edx/week-09-and-10-final-project"
path_dat <- file.path(path_raw, "data")
path_img <- file.path(path_raw, "presentation/img")
filename <- "dat_hr_all.feather"

options(tibble.width = Inf)

## define var for number of cpus and CV interations for parameter tuning:
## (benchmarking handled differently)
n_cpus <- 4
n_cv_iters_tuning <- 4

## decide what to do on learner errors (mlr package):
mlr::configureMlr(on.learner.error = "warn")

## ========================================================================= ##
## function definitions
## ========================================================================= ##

## customized ggsave function to avoid retyping all parameters:
ggsave_cust <- function(fname) 
  ggsave(filename = file.path(path_img, fname), 
         width = 8, height = 4, dpi = 200)

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
set.seed(452)
idx_train <- sample(1:nrow(dat_hr_mod), size = 26168, replace = FALSE) ## size = 26168 ## 1000 for testing
idx_test <- setdiff(1:nrow(dat_hr_mod), idx_train)
length(idx_train)

## don't save any objects until here:
obj_notsave <- ls()

## ========================================================================= ##
## restore everything from disk
## ========================================================================= ##

# load(file = file.path(path_dat, "kgl-mlr-trials_v001a.Rdata"))
# load(file = file.path(path_dat, "kgl-mlr-trials_v001b.Rdata"))

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
parallelStartMulticore(cpus = n_cpus, level = "mlr.resample")

## set random seed, also valid for parallel execution:
set.seed(4271, "L'Ecuyer")

## choose resampling strategy for parameter tuning:
rdesc <- makeResampleDesc(predict = "both", 
                          method = "CV", iters = n_cv_iters_tuning)
#method = "RepCV", reps = 3, folds = 5)

## parameters for parameter tuning:
ctrl <- makeTuneControlRandom(maxit = 40)
tune_measures <- list(rmse, mae, rsq, timetrain, timepredict)

## standard random forest implementation:
tic("time: tuning rf")
tune_results_rf <- tuneParams(
  "regr.randomForest", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
    makeIntegerParam("nodesize", lower = 10, upper = 50),
    makeIntegerParam("ntree", lower = 100, upper = 500)
  )
)
toc()
## time: tuning rf: 2275.833 sec elapsed (about 38 mins)

tune_results_rf
tune_results_rf$x

## faster random forest implementation:
tic("time: tuning ranger")
tune_results_ranger <- tuneParams(
  "regr.ranger", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
    makeIntegerParam("min.node.size", lower = 10, upper = 50),
    makeIntegerParam("num.trees", lower = 100, upper = 500)
  )
)
toc()
## time: tuning ranger: 508.617 sec elapsed (about 9 mins)
tune_results_ranger

## gradient boosting
tic("time: tuning gbm")
tune_results_gbm <- tuneParams(
  "regr.gbm", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("interaction.depth", lower = 1, upper = 9),
    makeIntegerParam("n.minobsinnode", lower = 10, upper = 50),
    makeIntegerParam("n.trees", lower = 100, upper = 1000)
  )
)
toc()
## time: tuning gbm: 309.933 sec elapsed (about 5 mins)

tune_results_gbm
#getParamSet("regr.gbm")

## gradient boosting using xgboost:
tic("time: tuning xgboost")
tune_results_xgboost <- tuneParams(
  "regr.xgboost", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("max_depth", lower = 1, upper = 9),
    makeIntegerParam("nrounds", lower = 100, upper = 1000)
    
  )
)
toc()
## time: tuning xgboost: 666.536 sec elapsed (about 11 mins)
tune_results_xgboost
#getParamSet("regr.xgboost")

parallelStop()

## ========================================================================= ##
## refit all learners with their tuned parameters (with CV)
## ========================================================================= ##

## start parallelization on benchmark level:
#parallelStartMulticore(cpus = n_cpus, level = "mlr.benchmark")
parallelStartMulticore(cpus = n_cpus, level = "mlr.resample")

## set random seed, also valid for parallel execution:
set.seed(427121, "L'Ecuyer")

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

## set resampling strategy for benchmarking:
rdesc_bm <- makeResampleDesc(predict = "both", 
                             method = "RepCV", reps = 3, folds = 4)

## refit tuned models on complete training data:
tic("time: refit tuned models on training data")
bmr_train <- benchmark(
  lrns_tuned, task, rdesc_bm,
  measures = list(rmse, #rmse.train.mean,
                  mae, #mae.train.mean,
                  rsq, #rsq.train.mean,
                  timetrain, timepredict)
)
toc()
## time: refit tuned models on training data: 890.584 sec elapsed (about 15 mins)
bmr_train

## visualizing benchmark results:
plotBMRBoxplots(bmr_train, measure = mae, style = "violin") +
  aes(fill = learner.id) + geom_point(alpha = .5)
plotBMRBoxplots(bmr_train, measure = timetrain, style = "violin") +
  aes(fill = learner.id) + geom_point(alpha = .5)


## ========================================================================= ##
## save snapshot to disk
## ========================================================================= ##

## save everything except data and path
obj <- setdiff(ls(), obj_notsave)
save(obj, file = file.path(path_dat, "kgl-mlr-trials_v001a.Rdata"))

## ========================================================================= ##
## use tuning wrappers in benchmark itself
## ========================================================================= ##

## also makes it possible to combine with different preprocessing...

## set random seed, also valid for parallel execution:
set.seed(427124, "L'Ecuyer")

## parameters for parameter tuning (search strategy and iterations):
ctrl <- makeTuneControlRandom(maxit = 40)
tune_measures <- list(rmse, mae, rsq, timetrain, timepredict)

## choose resampling strategy for parameter tuning:
rdesc_tune <- makeResampleDesc(predict = "both", 
                               method = "CV", iters = n_cv_iters_tuning)
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

## resamplnig strategy for benchmarking:
rdesc_bm <- mlr::makeResampleDesc(predict = "both", 
                               method = "RepCV", reps = 3, folds = 4)

#mlr::listMeasures(task)
tic("time: tuning and fitting to training data using wrappers")
bmr_tunewrap <- benchmark(
  lrns_tunewrap, task, rdesc_bm,
  # measures = list(rmse, mae, rsq)
  measures = list(rmse, #rmse.train.mean,
                  mae, #mae.train.mean,
                  rsq, #rsq.train.mean)
                  timetrain, timepredict)
)
toc()
## time: tuning and fitting to training data using wrappers: 22206.947 sec elapsed (about 370 mins = 6.2 hrs)
## note: individual tuning + refitting with RepCV: 2275 + 508 + 309 + 666 + 890 = 4648 secs
bmr_tunewrap


## visualizing benchmark results:
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
tic("time: refit models on complete training data, validate on test data")
bmr_full <- benchmark(
  lrns_tuned, task_full, rdesc_bmf,
  # measures = list(rmse, mae, rsq)
  measures = list(rmse, #rmse.train.mean,
                  mae, #mae.train.mean,
                  rsq, #rsq.train.mean,
                  timetrain, timepredict)
)
toc()
## time: refit models on complete training data, validate on test data: 191.371 sec elapsed (about 3.2 mins)
bmr_full

save(obj, file = file.path(path_dat, "kgl-mlr-trials_v001b.Rdata"))
parallelStop()




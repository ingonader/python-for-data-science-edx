## ######################################################################### ##
## Redo scikit-learn ML models from EdX-course 
## with R's mlr package
## ######################################################################### ##

# parallelMap::parallelStop()
# rm(list = ls(), inherits = TRUE); rstudioapi::restartSession()

## [[todo]]
## * on.learner.error option
## * use 4-fold CV in parameter tuning for comparison reasons (with python code)
## * add timing information to presentation

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
ggsave_cust <- function(fname, ...) 
  ggsave(filename = file.path(path_img, fname), 
         width = 8, height = 4, dpi = 200, ...)

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
## single learner without tuning (not used any further)
## ========================================================================= ##

## define learner without basic parameters:
learner_rf <- makeLearner("regr.randomForest", 
                          par.vals = list(
                            ntree = 500)
                          )
getParamSet("regr.randomForest")
getParamSet(learner_rf)

## get list of learners:
listLearners(warn.missing.packages = FALSE)
listLearners("regr", warn.missing.packages = FALSE)
listLearners("regr", properties = c("missings", "weights", "ordered"))

## see also:
## https://mlr.mlr-org.com/articles/tutorial/integrated_learners.html

## train learner:
model <- train(learner = learner_rf, task = task_full, subset = idx_train)
model

getLearnerModel(model) %>% class()
getLearnerModel(model)

## predict with learner:
pred <- predict(model, newdata = dat_hr_mod, subset = idx_test)
pred


## inspect predictions on a small subset
set.seed(1548)
task_small_prelim <- subsetTask(
  task = task_full, 
  subset = sample(idx_test, size = 1000)
)
plotLearnerPrediction(
  learner_rf, task = task_small_prelim, 
  features = "temp"
)
ggsave_cust("plot-learner-pred-1d.jpg")
plotLearnerPrediction(
  learner_rf,
  task = task_small_prelim, features = c("temp", "rel_hum")
)
ggsave_cust("plot-learner-pred-2d.jpg")

## assess performance of learner:
performance(pred, measures = list(mse, mae, rsq))

## list of suitable measures:
listMeasures()
listMeasures("regr")
listMeasures(task)  ## for a specific task

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

## not needed: estimating performance using resampling:
res <- resample(learner = "regr.ranger", task = task, 
                resampling = rdesc,
                measures = list(mse, mae, rsq))

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
## time: tuning rf: 2275.833 sec elapsed with 3-fold CV (about 38 mins)
## time: tuning rf: 2861.12 sec elapsed with 4-fold CV (about 48 mins) with lots of other apps open
## time: tuning rf: 1979.765 sec elapsed with 4-fold CV (about 33 mins)
## time: tuning rf: ? sec elapsed with 4-fold CV



getParamSet("regr.randomForest")
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
## time: tuning ranger: 508.617 sec elapsed with 3-fold CV (about 9 mins)
## time: tuning ranger: 960.068 sec elapsed with 4-fold CV (about 16 mins) with lots of other apps open
## time: tuning ranger: 646.674 sec elapsed with 4-fold CV (about 11 mins)
## time: tuning ranger: 674.228 sec elapsed with 4-fold CV
tune_results_ranger

## gradient boosting via gbm
tic("time: tuning gbm")
tune_results_gbm <- tuneParams(
  "regr.gbm", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeDiscreteParam("distribution", c("laplace")),
    makeNumericParam("shrinkage", lower = 0.01, upper = .2),
    makeIntegerParam("interaction.depth", lower = 5, upper = 25),
    makeIntegerParam("n.minobsinnode", lower = 10, upper = 35),
    makeIntegerParam("n.trees", lower = 2000, upper = 10000)
  )
)
toc()
## time: tuning gbm: 309.933 sec elapsed with 3-fold CV (about 5 mins)
## time: tuning gbm: 587.238 sec elapsed with 4-fold CV (about 10 mins) with lots of other apps open
## time: tuning gbm: 15736.575 sec elapsed with 4-fold CV (about 262 mins = 4.3 hrs) with up to 10000 trees
## time: tuning gbm: 15550.366 sec elapsed with 4-fold CV

## gradient boosting via gbm (modified)
tic("time: tuning gbm modified")
tune_results_gbm_mod <- tuneParams(
  "regr.gbm", 
  task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeDiscreteParam("distribution", c("laplace")),
    makeNumericParam("shrinkage", lower = 0.1, upper = .6),
    makeIntegerParam("interaction.depth", lower = 5, upper = 20),
    makeIntegerParam("n.minobsinnode", lower = 10, upper = 35),
    makeIntegerParam("n.trees", lower = 500, upper = 1500)
  )
)
toc()
## time: tuning gbm modified: 2175.284 sec elapsed with 4-fold CV (about 36 mins)


tune_results_gbm_mod
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
## time: tuning xgboost: 666.536 sec elapsed with 3-fold CV (about 11 mins)
## time: tuning xgboost: 738.054 sec elapsed with 4-fold CV (about 12 mins) with lots of other apps open
## time: tuning xgboost: 830.326 sec elapsed with 4-fold CV (about 14 mins)
## time: tuning xgboost: 708.274 sec elapsed with 4-fold CV (about 12 mins)

tune_results_xgboost
#getParamSet("regr.xgboost")


# ## gradient boosting via bst
# tic("time: tuning bst")
# tune_results_bst <- tuneParams(
#   "regr.bst", 
#   task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
#   #par.vals = list(xval = 4),
#   par.set = makeParamSet(
#     makeIntegerParam("xval", lower = 4, upper = 4),  ## internal cross-validation
#     makeIntegerParam("mstop", lower = 50, upper = 201), ## number of boosting interations
#     makeNumericParam("nu", lower = .05, upper = .2),    ## step size or shrinkage parameter [[?]]
#     makeIntegerParam("minbucket", lower = 30, upper = 60),
#     makeIntegerParam("maxdepth", lower = 4, upper = 21)
#   )
# )
# toc()
# ## time: tuning bst: 267.062 sec elapsed with 4-fold CV (about 5 mins) with lots of other apps open
# ## note: sucks on small sample sizes; and on larger sample sizes as well?
# 
# tune_results_bst
# #getParamSet("regr.bst")
# 
# ## gradient boosting via glmboost
# tic("time: tuning glmboost")
# tune_results_glmboost <- tuneParams(
#   "regr.glmboost",
#   task = task, resampling = rdesc, measures = tune_measures, control = ctrl,
#   par.set = makeParamSet(
#     makeDiscreteParam("family", c("Gaussian", "Huber", "Poisson")),
#     makeLogicalParam("center", TRUE),
#     makeIntegerParam("mstop", lower = 50, upper = 201), ## number of boosting interations [[?]]
#     makeNumericParam("nu", lower = .05, upper = .2)    ## step size or shrinkage parameter [[?]]
#   )
# )
# toc()
# ##time: tuning glmboost: 55.027 sec elapsed with 4-fold CV with lots of other apps open
# ## note: sucks on small sample sizes; and on larger sample sizes as well?
# 
# tune_results_glmboost
# #getParamSet("regr.glmboost")


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
  makeLearner("regr.randomForest", par.vals = tune_results_rf$x),
  makeLearner("regr.ranger", par.vals = tune_results_ranger$x),
  makeLearner("regr.gbm", par.vals = tune_results_gbm_mod$x),
  makeLearner("regr.gbm", id = "regr.gbm.ntreeplus", par.vals = tune_results_gbm$x),
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
## time: refit tuned models on training data: 954.877 sec elapsed (about 16 mins)
## time: refit tuned models on training data: 2594.893 sec elapsed (about 43 mins)

bmr_train

## visualizing benchmark results:
plotBMRBoxplots(bmr_train, measure = mae, style = "violin") +
  aes(fill = learner.id) + geom_point(alpha = .5)
ggsave_cust("plot-bmr-boxplot-mae.jpg", scale = .75)

plotBMRBoxplots(bmr_train, measure = timetrain, style = "violin") +
  aes(fill = learner.id) + geom_point(alpha = .5)


## ========================================================================= ##
## save snapshot to disk
## ========================================================================= ##

## save everything except data and path
obj <- setdiff(ls(), obj_notsave)
save(list = obj, file = file.path(path_dat, "kgl-mlr-trials_v001a.Rdata"))

## ========================================================================= ##
## use tuning wrappers in benchmark itself
## ========================================================================= ##

parallelStartMulticore(cpus = n_cpus, level = "mlr.resample")

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
    makeDiscreteParam("distribution", c("laplace")),
    makeNumericParam("shrinkage", lower = 0.01, upper = .2),
    makeIntegerParam("interaction.depth", lower = 5, upper = 25),
    makeIntegerParam("n.minobsinnode", lower = 10, upper = 35),
    makeIntegerParam("n.trees", lower = 2000, upper = 7000) ## 10000 above?
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

# #mlr::listMeasures(task)
# tic("time: tuning and fitting to training data using wrappers")
# bmr_tunewrap <- benchmark(
#   lrns_tunewrap, task, rdesc_bm,
#   # measures = list(rmse, mae, rsq)
#   measures = list(rmse, #rmse.train.mean,
#                   mae, #mae.train.mean,
#                   rsq, #rsq.train.mean)
#                   timetrain, timepredict)
# )
# toc()
# ## time: tuning and fitting to training data using wrappers: 22206.947 sec elapsed (about 370 mins = 6.2 hrs)
# ##       note: individual tuning + refitting with RepCV: 2275 + 508 + 309 + 666 + 890 = 4648 secs
# ## time: tuning and fitting to training data using wrappers: 31991.86 sec elapsed (about 8.9 hrs)
# ##       (but not finished yet ... cancelled by user)
# 
# bmr_tunewrap
#
#
# ## visualizing benchmark results:
# plotBMRBoxplots(bmr_tunewrap, measure = mae, style = "violin") +
#   aes(fill = learner.id) + geom_point(alpha = .5)


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
## time: refit models on complete training data, validate on test data: 238.926 sec elapsed (about 4 mins)

bmr_full

parallelStop()


## [[here]]
## [[todo]]
## score best model on test data, without refitting? (for python comparision)


## ========================================================================= ##
## save snapshot to disk
## ========================================================================= ##

## save everything except data and path
obj <- setdiff(ls(), obj_notsave)
save(list = obj, file = file.path(path_dat, "kgl-mlr-trials_v001b.Rdata"))

## ========================================================================= ##
## inspect best model predictions
## ========================================================================= ##

# bmr_full$results$trip_cnt_mod$regr.xgboost %>% str(max.level = 1)
# getBMRPredictions(bmr_full)$trip_cnt_mod$regr.xgboost
# varnames_features


# ## get learner:
# getBMRLearners(bmr_full)$regr.xgboost

# ## get model:
# getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]]$learner.model

## make smaller task (data subset) 
## for faster inspection:
set.seed(1548)
task_small <- subsetTask(task = task_full, 
                         subset = sample(idx_test, size = 1000))

plotLearnerPrediction(
  getBMRLearners(bmr_full)$regr.xgboost,
  task = task_small, features = "temp"
)

plotLearnerPrediction(
  getBMRLearners(bmr_full)$regr.xgboost,
  task = task_small, features = c("temp", "rel_hum")
)

## access models of original packages:
getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]]$learner.model
getBMRModels(bmr_full)$trip_cnt_mod$regr.randomForest[[1]]$learner.model

## acces model wrappers (mlr wrappers):
getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]]
getBMRModels(bmr_full)$trip_cnt_mod$regr.randomForest[[1]]

## ========================================================================= ##
## inspect best model using iml package
## ========================================================================= ##

library(iml)

## take sample for quicker model exploration:
set.seed(442)
dat_iml <- dat_hr_mod[idx_test, ] %>% sample_n(500)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## feature importance: main effects and interactions
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## create a predictor container(s):
predictor_rf <- Predictor$new(
  model = getBMRModels(bmr_full)$trip_cnt_mod$regr.randomForest[[1]],
  data = dat_iml[varnames_features],  y = dat_iml[varnames_target]
)
predictor_ranger <- Predictor$new(
  model = getBMRModels(bmr_full)$trip_cnt_mod$regr.ranger[[1]],
  data = dat_iml[varnames_features],  y = dat_iml[varnames_target]
)
predictor_gbm <- Predictor$new(
  model = getBMRModels(bmr_full)$trip_cnt_mod$regr.gbm[[1]],
  data = dat_iml[varnames_features],  y = dat_iml[varnames_target]
)
predictor_xgboost <- Predictor$new(
  model = getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]],
  data = dat_iml[varnames_features],  y = dat_iml[varnames_target]
)
## "choose" a standard predictor to be used below:
predictor <- predictor_xgboost

varimp_scale <- .8

## most important features:
imp_xgboost <- FeatureImp$new(predictor_xgboost, loss = "mae")
plot(imp_xgboost)
ggsave_cust("plot-varimp-xgboost.jpg", scale = varimp_scale)

imp_rf <- FeatureImp$new(predictor_rf, loss = "mae")
plot(imp_rf)
ggsave_cust("plot-varimp-rf.jpg", scale = varimp_scale)


## most important interactions:
## plots how much of the variance of f(x) is explained by the interaction. 
## The measure is between 0 (no interaction) and 1 (= 100% of variance of f(x) 
## due to interactions)
interact_xgboost <- Interaction$new(predictor_xgboost)
plot(interact_xgboost)
ggsave_cust("plot-varimp-interact-xgboost.jpg", scale = varimp_scale)

interact_rf <- Interaction$new(predictor_rf)
plot(interact_rf)
ggsave_cust("plot-varimp-interact-rf.jpg", scale = varimp_scale)

## most important interactions:
interact_hr_of_day <- Interaction$new(predictor, feature = "hr_of_day")
plot(interact_hr_of_day)
ggsave_cust("plot-varimp-interact-xgboost-hr_of_day.jpg", scale = varimp_scale)
interact_temp <- Interaction$new(predictor, feature = "temp")
plot(interact_temp)
ggsave_cust("plot-varimp-interact-xgboost-temp.jpg", scale = varimp_scale)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## feature effects (with iml)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


eff_scale <- .8

# ## accumulated local effects (ALE) for specific feature:
# # (similar to partial dependence plots):
effs <- FeatureEffect$new(predictor, feature = "temp")
plot(effs)
ggsave_cust("plot-effects-temp-ale.jpg", scale = eff_scale)

## partial dependence plot with ice plot:
effs <- FeatureEffect$new(predictor, feature = "temp", method = "pdp+ice")
plot(effs)
ggsave_cust("plot-effects-temp-ice.jpg", scale = eff_scale)

# ## accumulated local effects for all features (quartz device):
# # (similar to partial dependence plots):
# effs <- FeatureEffects$new(predictor)
# plot(effs)
# 
# ## partial dependence and ice plots for all features (quartz device):
# effs <- FeatureEffects$new(predictor, method = "pdp+ice")
# plot(effs)

## pdp of interactions
tic("time: feature interaction plot")
effs_int <- FeatureEffect$new(predictor, 
                              feature = c("temp", "hr_of_day"), 
                              method = "pdp", grid.size = 40)  ## use "ale"?
toc()
## time: feature interaction plot:    0.79 sec elapsed (method = "ale", grid.size = 40, sample_n(500))
## time: feature interaction plot: 179.153 sec elapsed (method = "pdp", grid.size = 40, sample_n(500))
plot(effs_int)

effs_int <- FeatureEffect$new(predictor, 
                              feature = c("hr_of_day", "day_of_week"), 
                              method = "pdp", grid.size = 40)  ## use "ale"?
plot(effs_int)

## ========================================================================= ##
## inspect model using pdp package (and maye ICEbox)
## ========================================================================= ##

library(pdp)

## ice object for ice plot (single continuous variable):
tic("time: ice object")
ice_object <- pdp::partial(
  object = getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]], 
  train = dat_iml[varnames_model] %>% as.data.frame(),
  type = "regression",
  ice = TRUE,
  pred.var = "temp",
  pred.fun = function(object, newdata) {
    pred_mlr <- predict(
      getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]], 
      newdata = newdata)
    ret <- getPredictionResponse(pred_mlr)
    return(ret)
  },
  grid.resolution = 20
)
toc()
## time: ice object: 3.432 sec elapsed (grid.resolution = 20, n_sample(500))
## individual conditional independent expectations (ice):
autoplot(
  ice_object, alpha = .2, rug = TRUE,
  train = dat_iml, size = .5, color = "grey"
)

## ice object for pdp intraction plot (2 cont. variables):
tic("time: ice object for pdp intraction")
ice_object_int <- partial(
  object = getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]], 
  train = dat_iml[varnames_model] %>% as.data.frame(),
  #type = "regression",
  chull = TRUE,
  #pred.var = c("temp", "rel_hum"),
  pred.var = c("hr_of_day", "day_of_week"),
  pred.fun = function(object, newdata) {
    pred_mlr <- predict(
      getBMRModels(bmr_full)$trip_cnt_mod$regr.xgboost[[1]],
      newdata = newdata)
    ret <- getPredictionResponse(pred_mlr)
    return(ret)
  },
  grid.resolution = 40
)
toc()
## time: ice object for pdp intraction: 26.518 sec elapsed (grid.resolution = 20, n_sample(500))

## [[note]]
## for some reason, specifying the prediction function always returns an 
## ice object (class c("data.frame", "ice")), which has an predictions for
## each case (and an yhat.id column).-
## to plot partial dependence polts in 2D, this class should be 
## c("data.frame", "partial"), and that column has to be removed (average).
class(ice_object_int)
head(ice_object_int)
dim(ice_object_int)

## some manual tweaking:
ice_object_int <- ice_object_int %>% group_by_at(c(1, 2)) %>% summarize(yhat = mean(yhat)) %>% as.data.frame()
class(ice_object_int) <- c("data.frame", "partial")

## plot:
#rwb <- colorRampPalette(c("red", "white", "blue"))
pdp::plotPartial(
  ice_object_int, levelplot = TRUE, contour = TRUE, chull = TRUE,
  train = dat_iml[varnames_model] %>% as.data.frame() ## (convec hull needs training data)
  #col.regions = rwb
)
#pdp::plotPartial(ice_object_int)

## with ggplot:
autoplot(ice_object_int, contour = TRUE, legend.title = "Partial\ndependence")



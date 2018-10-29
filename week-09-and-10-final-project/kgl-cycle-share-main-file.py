## ######################################################################### ##
## Analysis of 
## For EdX Course
## Python for Data Science (Week 9 and 10 Final Project)
## ######################################################################### ##

## ========================================================================= ## 
## main structure of analysis
## ========================================================================= ##

exec(open("./kgl-cycle-share-01-setup.py").read())  ## no magic allowed
exec(open("./kgl-cycle-share-02-data-download.py").read())
exec(open("./kgl-cycle-share-03-data-load.py").read())
exec(open("./kgl-cycle-share-04-data-prep.py").read())

#%run -i kgl-cycle-share-05-exploratory-analysis.py

## open in editor, as execution makes no sense (and also, contains magic)
exec(open("./kgl-cycle-share-05-exploratory-analysis.py").read())

## open any of the 06<x> script to estimate and save a model:
# exec(open("./kgl-cycle-share-06a-random-forest.py").read())

## open the 07-eval-model.py script, choose a model and produce plots:
# exec(open("./kgl-cycle-share-07-eval-model.py").read())




## [[todo]] 
## * build a pipeline to score classification, then regression model in one action
## * rerun the whole analysis in a pipenv, and store requirements.txt on github
## * build a notebook with the most important parts of the code (along with the final presentation)



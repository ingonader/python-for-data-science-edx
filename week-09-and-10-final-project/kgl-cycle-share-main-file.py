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

## open in editor, as execution makes no sense (and also, contains magic)
exec(open("./kgl-cycle-share-05-exploratory-analysis.py").read())

exec(open("./kgl-cycle-share-06a-random-forest.py").read())
exec(open("./kgl-cycle-share-main-file.py").read())


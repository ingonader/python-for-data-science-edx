## ######################################################################### ##
## Analysis of Kaggle Cycle Share Data 
## https://www.kaggle.com/pronto/cycle-share-dataset/home
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
from plotnine import *
import matplotlib.pyplot as plt



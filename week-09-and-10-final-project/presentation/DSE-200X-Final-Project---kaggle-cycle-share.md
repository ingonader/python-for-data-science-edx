---
title: "Influences on Bike Rentals"
subtitle: "What are the influening factors and <br>how well can they predict the number of rented bikes?"
author: "Ingo Nader"
date: "Oct 2018"
#output: html_document
output: 
  ioslides_presentation:
    css: styles-edx.css
    #logo: img/logo-um-154x127.png
    widescreen: true
    keep_md: true
    #smaller: true  ## only works without "---" slide breaks (use ##)
    slide_level: 2
## Comments and Instructions
##
## ## ------------------------------------------- ##
## ## Controlling presentation (best use chrome):
## ## ------------------------------------------- ##
    # 'f' enable fullscreen mode
    # 'w' toggle widescreen mode
    # 'o' enable overview mode
    # 'h' enable code highlight mode
    # 'p' show presenter notes
##
## ## ------------------------------------------- ##
## ## Images
## ## ------------------------------------------- ##
##
## Replace markdown images "![]()" with R's include_graphics()
## (in order for them to scale to slide width properly):
## Search:
## !\[\]\((.*)\)
## Replace with:
## ```{r, eval = TRUE, echo = FALSE, out.width = "100%", fig.align = "left"}\nknitr::include_graphics("\1")\n```
##
##
## ## ------------------------------------------- ##
## ## Font size in slides, and other layout stuff
## ## ------------------------------------------- ##
##
## use {.smaller} after title for single slides
## use {.flexbox .vcenter} for centering of text
## 
## ## ------------------------------------------- ##
## ## color:
## ## ------------------------------------------- ##
##
##   <div class="red2"></div>
## or:
##   <font color="red"> </font>
##
## ## ------------------------------------------- ##
## ## two-column layout:
## ## ------------------------------------------- ##
## 
## <div></div><!-- ------------------------------- needed as is before cols - -->
## <div style="float: left; width: 49%;"><!-- ---- start of first column ---- -->
## Put col 1 markdown here
## </div><!-- ------------------------------------ end of first column ------ -->
## <div style="float: left; width: 2%"><br></div><!-- spacing column -------- -->
## <div style="float: left; width: 49%;"><!-- ---- start of second column --- --> 
## Put col 2 markdown here
## </div><!-- ------------------------------------ end of second column ----- -->
## <div style="clear: both"></div><!-- end cols for text over both cols below -->
##
## additionally, if one column needs to start higher (for right columns and 
## short slide titles, mostly):
## <div style="float: left; width: 30%; margin-top: -15%"><!-- ---- start of second column              --> 
## 
## other possibilities (not as good):
## * In slide title line, use:
##   ## title {.columns-2}
## * put md into this div:
##   <div class="columns-2">  </div>
##
---
[//]: # (
http://www.w3schools.com/css/css_font.asp
http://www.cssfontstack.com/Helvetica
)

<style>
/* gdbar size (that contains logo) on title page */
/* needs to have greater height than logo image, other stuff is irrelevant */
.gdbar {
  position:absolute !important;
  top: 50px !important; left: auto; right: 0px !important; width: 0px !important;
  height: 500px !important;  /* modify if logo is larger than this in height */
}

/* logo size on title page */
.gdbar img {
  position: absolute; 
  top: 0px;
  left: 50px;
  width: 154px !important;
  height: 127px !important;
}

/* logo size on slides */
slides > slide:not(.nobackground):before {
  width: 77px; height: 64px; /* modify width and height (twice) */
  background-size: 77px 64px;
  position: absolute; left: auto;
  right: -30px;  /* modify position */
  top: 10px;
}

/*slides > slide.backdrop {   */
/*  background-color:#ffaaaa;   */
/*  background:#ffaaaa;   */
/*}   */
</style>




## [[todo]]

* different font for latex formulas?

## Abstract

This piece of work investigates the influences of weather on bike rentals.
...[[todo]]

> Summarize your questions and findings in 1 brief paragraph (4-6 sentences max). Your abstract needs to include: what dataset, what question, what method was used, and findings.
 
## Motivation
Describe the problem you want to solve with the data. It may relate closely with your research question, but your goal here is to make your audience care about the project/problem you are trying to solve. You need to articulate the problem you are exploring and why (and for whom) insight would be valuable.
 
## Dataset(s) {.smaller}

Two datasets were used.

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 50%;"><!-- ---- start of first column               -->

Bike sharing data:

* BIXI Montreal public bicycle sharing system, North America's first 
  large-scale bike sharing system
* Available via kaggle from [https://www.kaggle.com/aubertsigouin/biximtl/home](https://www.kaggle.com/aubertsigouin/biximtl/home)
* For years 2014 to 2017
* Contains individual records of bike trips: timestamp and station code for 
  start and end of trip, duration
* $n = 14598961$ records (individual bike trips)
* Station codes, names, and position (latitude, longitude) 
  available in separate files, but only of secondary interest for this analysis

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 50%;"><!-- ---- start of second column              --> 

Weather data from the Canadian government:

* [http://climate.weather.gc.ca/historical_data/search_historic_data_e.html](http://climate.weather.gc.ca/historical_data/search_historic_data_e.html)
* API for bulk data download: [http://climate.weather.gc.ca/climate_data/bulk_data_e.html](http://climate.weather.gc.ca/climate_data/bulk_data_e.html)
* Data can be downloaded per weather station per month and contains 
  hourly measurements of different metrics (e.g., timestamp, temperature, 
  relative humidity, atmospheric pressure, wind speed; different measures 
  available for different stations)
* $n = 35064$ hourly weather records in total (between 672 and 744 per monthly file)
* List of available weather stations: [[?]]

</div><!-- ------------------------------------ end of second column                -->
<div style="clear: both"></div><!-- end floating section for text over both cols below -->

> Describe your dataset(s) here. You should say what data is in the dataset, how much data, and where you found the dataset (if applicable).


## Data Preparation and Cleaning

* First, data download was performed manually for the bike share data from kaggle (as only available after login), and via a Python script for the weather data (bulk download).
* Next, the data was loaded and contatenated into a pandas `DataFrame` each for individual bike rides and hourly weather data.
* The next step was calculating the variable of interest: Hourly bike rides. This was done by aggregating individual bike trips to hourly counts of trips (how many trips in each hour), using the starting time of the trip.
* Then, the weather data was joined to the hourly bike ride data, using the common timestamp.
* One feature (wind chill) was dropped, as it had too many missing values (77.9% missing).
* Finally, addtional features were added for the analysis: hour of the day (0-23), and day of the week (0-6, zero corresponding to Monday, six corresponding to Sunday).
* For modeling, rows with missing values were dropped, as the goal is not having the most complete prediction coverage, but rather an indication of the prediction quality that is possible with complete data. In total, 1284 rows (0.04%) of the original data were dropped.
* The remaining rows were split into training set (90% of the data $n = 26168$ rows) and testing set (the remaining 10%, $n = 2908$).

* [[?]] dewpoint feature?

## Research Question(s)

The research questions that I wanted to answer with my analysis were:

* To what extent do the number of bike rides depend on the current weather conditions? That is, how well can the number of bike rides be predicted from weather data (and time of year, time of day)?
* What are the most important factors that influence the number of bike rides?
* How do these factors influence the number of bike rides?


## Methods

First, some data exploration was performed, in order to get to know the data 
and to find out how the number of hourly bike trips is distributed across the 
investigated time span. Also, the interrelation of features was investigated
by means of a correlation heatmap.

In order to find out how well the number of bike rides can be predicted 
from the data, different approaches were taken. As a baseline model, a rolling
mean was calculated to find out how this very simple model can explain the data.

Then, after splitting the data into 90% training and 10% test set, 
different machine learning models were fitted to the data in order to predict 
the hourly number of bike rides from the available data: Random forest regression, 
and gradient boosting regression via `scikit-learn` and `xgboost`. The most 
promising model, `scikit-learn`'s gradient boosting regression, was fitted
via a randomized 4-fold cross-validation for indentifying the best hyperparameters.
Variable importance was used to identify the most important influence factors,
and partial dependence plots (PDP) and Individual Conditional Expectation (ICE)
plots were used to visualize the influences of the important variables on the 
number of bike trips.



[[todo]]
* Reference for scikit-learn GBR, Variable iMportance, pdp and ice plots


> What methods did you use to analyze the data and why are they appropriate? Be sure to adequately, but briefly, describe your methods.


## Findings: Data Exploration

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 55%"><!-- ---- start of first column               -->

To get a better understanding of the data, the number of hourly bike trips
was visualized for the time span between 2014 and 2017. 

The moving average
that is shown in the plot (red line) can be interpreted as a *baseline model*, i.e., 
the simplest possible model to describe the hourly number of bike rides. 
This baseline model explains 38.8% of the variance $(r^2 = 0.388)$ and has
a mean absolute error of 316.2, which means that on average, the "prediction" 
for the number of hourly bike rides is wrong by this many bike rides.

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 1%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 44%; margin-top: -3%"><!-- ---- start of second column              --> 
<img src="img/expl-trips-per-hour-2014-2017.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: Number of hourly rides from 2014 to 2017. Each dot represents the 
number of trips in one specifc hour. Red line represents a 
moving average using a window of 14 days.
<p>

</div><!-- ------------------------------------ end of second column                -->


## Findings: Exploration

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 55%"><!-- ---- start of first column               -->

To visualize the relations between the available features,
a correlation heatmap is shown on the right. The features are only
slightly correlated, with the only exception being temperature and dew point
that show an almost perfect (linear) relationship $(r = .93)$.

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 1%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 44%; margin-top: 0%"><!-- ---- start of second column              --> 
<img src="img/expl-corr-heatmap.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: Pearson Correlations between available features in the data.
<p>

</div><!-- ------------------------------------ end of second column                -->




## Findings

Findings Outline:

* Data Exploration: Mean as baseline model?
* Correlation map?
* Model and Model Quality (+ pic predicted vs. actual), MAE, r^2 
    * numerical features only, also categorical models tried (but not as good)
* Important variables
* PDP for important variables
* Interaction Plots for important variables

>Feel free to replicate this slide to show multiple findings
>Present your findings. Include at least one visualization in your presentation (feel free to include more). The visualization should be honest, accessible, and elegant for a general audience.
You need not come to a definitive conclusion, but you need to say how your findings relate back to your research question.


 
## Limitations

* Results only valid for cities with roughly the same climate.
* Bike rides might be influenced by weather *predictions* for a given day, not only by the actual weather.
* Precipitation (rain, snow) might also be a very good predictor; unfortunately, this was not easily available for the given weather station.
* Maybe have slightly overfitted the training data, but still a good result (MAE).
* Missing values: ignored / imputed?
* Using time and month also, or only weather data?

> If applicable, describe limitations to your findings. For example, you might note that these results were true for British Premier league players but may not be applicable to other leagues because of differences in league structures.
> Or you may note that your data has inherent limitations. For example, you may not have access to the number of Twitter followers per users so you assumed all users are equally influential. If you had the number of followers, you could weight the impact of their tweet’s sentiment by their influence (# of followers).

## Conclusions

> Report your overall conclusions, preferably a conclusion per research question
 
## Acknowledgements
Where did you get your data? Did you use other informal analysis to inform your work? Did you get feedback on your work by friends or colleagues? Etc. If you had no one give you feedback and you collected the data yourself, say so.
 
## References
If applicable, report any references you used in your work. For example, you may have used a research paper from X to help guide your analysis. You should cite that work here. If you did all the work on your own, please state this.
 

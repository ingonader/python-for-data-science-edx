---
title: "Weather and Bike Rentals"
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
csl: plos-one.csl
references:
- id: rf_imp
  title: Beware Default Random Forest Importances
  author:
  - family: Parr
    given: Terence
  - family: Turgutlu
    given: Kerem
  - family: Csiszar
    given: Christopher
  - family: Howard
    given: Jeremy
  URL: 'http://explained.ai/rf-importance/index.html'
  issued:
    year: 2018
    month: 3
- id: rf_or_gbm
  title: Random forest or gradient boosting?
  author: 
  - family: Wheatley
    given: Joe
  URL: 'http://joewheatley.net/random-forest-or-gradient-boosting/' 
  issued:
    year: 2014
    month: 2
- id: ice_plots
  title: "Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation"
  author:
    - family: Goldstein
      given: Alex
    - family: Kapelner
      given: Adam
    - family: Bleich
      given: Justin
    - family: Pitkin
      given: Emil
  URL: 'https://arxiv.org/pdf/1309.6392.pdf'
  issued:
    year: 2014
    month: 3
- id: pressure_and_rain
  title: "Why Does it Rain When the Pressure Is Low?"
  author:
    - family: Morgan
      given: Lee
  URL: 'https://sciencing.com/rain-pressure-low-8738476.html'
  issued: 
    year: 2017
    month: 4
- id: humidity_and_rain
  title: "RELATIVE HUMIDITY PITFALLS"
  author:
    - family: Haby
      given: Jeff
  URL: 'http://www.theweatherprediction.com/habyhints2/564/'
- id: humidity_and_temp
  title: "How Temperature & Humidity are Related"
  author:
    - family: Dotson
      given: J. Dianne
  URL: 'https://sciencing.com/temperature-ampamp-humidity-related-7245642.html'
  issued:
    year: 2018
    month: 4
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
## <div style="float: left; width: 48%;"><!-- ---- start of first column ---- -->
## Put col 1 markdown here
## </div><!-- ------------------------------------ end of first column ------ -->
## <div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
## <div style="float: left; width: 48%;"><!-- ---- start of second column --- --> 
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
/* justify text: */
body {
  text-align: justify
}

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

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  jax: ["input/TeX", "output/HTML-CSS"],
  "HTML-CSS": { 
      preferredFont: "Arial", 
      availableFonts: [],
      scale: 85
      // styles: {".MathJax": {color: "#CCCCCC"}} 
      }
});
</script>





## [[todo]]

* Figure and Table numbering? --> rather not.
* Figure caption in first column, for ICE plots?
* Or figure caption in separate column with font facing upwards (90% rotated)?
* Features: numeric, not categorical (!)

## Abstract

This piece of work investigates the influences of weather on bike rentals.
...[[todo]]

> Summarize your questions and findings in 1 brief paragraph (4-6 sentences max). Your abstract needs to include: what dataset, what question, what method was used, and findings.
 
## Motivation
Describe the problem you want to solve with the data. It may relate closely with your research question, but your goal here is to make your audience care about the project/problem you are trying to solve. You need to articulate the problem you are exploring and why (and for whom) insight would be valuable.
 
## Dataset(s) {.smaller}


<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 48%;"><!-- ---- start of first column               -->

Two datasets were used: **Bike sharing data**...

* **BIXI Montreal public bicycle sharing system**, North America's first 
  large-scale bike sharing system
* Available via kaggle from<br> [https://www.kaggle.com/aubertsigouin/biximtl/home](https://www.kaggle.com/aubertsigouin/biximtl/home)
* For years $2014$ to $2017$
* Contains **individual records of bike trips**: timestamp and station code for 
  start and end of trip, duration
* $n = 14598961$ records (individual bike trips)
* Station codes, names, and position (latitude, longitude) 
  available in separate files, but only of secondary interest for this analysis

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
<div style="float: left; width: 48%;"><!-- ---- start of second column              --> 

...and **weather data** from the Canadian government:

* Available from<br> [http://climate.weather.gc.ca/ 
  historical_data/ <br> search_historic_data_e.html](http://climate.weather.gc.ca/historical_data/search_historic_data_e.html)
* API for **bulk data download**:<br> [http://climate.weather.gc.ca/climate_data/ <br> bulk_data_e.html](http://climate.weather.gc.ca/climate_data/bulk_data_e.html)
* Data can be downloaded per weather station per month and contains 
  **hourly measurements** of different metrics (e.g., timestamp, temperature, 
  relative humidity, atmospheric pressure, wind speed; different measures 
  available for different stations)
* $n = 35064$ hourly weather records in total (between $672$ and $744$ per monthly file)

</div><!-- ------------------------------------ end of second column                -->
<div style="clear: both"></div><!-- end floating section for text over both cols below -->


## Data Preparation and Cleaning {.smaller}

<div></div><!-- ------------------------------- needed as is before cols - -->
<div style="float: left; width: 48%;"><!-- ---- start of first column ---- -->

* First, **data download** was performed manually for the bike share data 
  from kaggle (as only available after login), and via a Python script 
  for the weather data (bulk download).
* For the weather data, the **weather station** that was most central to the
  locations of the bike rides was picked (see data exploration).
* Next, the **data was loaded** and contatenated into a pandas `DataFrame` 
  each for individual bike rides and hourly weather data.
* The next step was **calculating the variable of interest: Hourly bike rides**. 
  This was done by aggregating individual bike trips to hourly counts of 
  trips (how many trips in each hour), using the starting time of the trip.
* Then, the **weather data was joined to the hourly bike ride data**, 
  using the common timestamp as join key.

</div><!-- ------------------------------------ end of first column ------ -->
<div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
<div style="float: left; width: 48%;"><!-- ---- start of second column --- --> 

* One feature **(wind chill) was dropped**, as it had too many 
  missing values ($77.9\%$ missing).
* Finally, **addtional features were added** for the analysis: 
  hour of the day ($0$-$23$), and day of the week ($0$-$6$, zero corresponding 
  to Monday, six corresponding to Sunday).
* For modeling, **rows with missing values were dropped**, as the goal 
  is not having the most complete prediction coverage, but rather an 
  indication of the prediction quality that is possible with complete data. 
  In total, $1284$ rows ($0.04\%$) of the original data were dropped.
* The remaining rows were **split into training and testing set** ($90\%$ of the data, 
  $n = 26168$ rows for training, the remaining $10\%$, $n = 2908$ for testing).

</div><!-- ------------------------------------ end of second column ----- -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


## Research Question(s)

The research questions that I wanted to answer with my analysis were:

* **To what extent do** the number of **bike rides depend on the current 
  weather conditions**? That is, how well can the number of bike rides 
  be predicted from weather data (and time of year, time of day)?
* What are the **most important factors** that influence the number 
  of bike rides?
* **How do these factors influence** the number of bike rides? What are 
  the main effects of these factors, and what are the interactions between
  them?


## Methods

First, some **data exploration** was performed, in order to get to know the data 
and to find out how the number of hourly bike trips is distributed across the 
investigated time span. Also, the interrelation of features was investigated
by means of a correlation heatmap.

In order to find out how well the number of bike rides can be predicted 
from the data, different models were used As a **baseline model**, a moving
average was calculated to find out how this very simple model can explain the data.

Then, after splitting the data into $90\%$ training and $10\%$ test set, 
**different machine learning models** were fitted to the data in order to predict 
the hourly number of bike rides from the available data: Random forest regression, 
and gradient boosting regression via `scikit-learn` and `xgboost`. The most 
promising model, `scikit-learn`'s gradient boosting regression, was fitted
via a randomized $4$-fold cross-validation for indentifying the best hyperparameters.
Variable importance was used to identify the most important influence factors,
and *Partial Dependence Plots* (*PDP*) and *Individual Conditional Expectation* (*ICE*)
plots [@ice_plots] were used to 
**visualize the influences of the important variables** 
on the number of bike trips.


## Findings: Data Exploration


<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 53%"><!-- ---- start of first column               -->

**Choosing the closest weather station:**

The Canadian government's past weather and climate service offers a 
*search by proximity* function. Via this service, some sample data 
of the closest stations to
Montreal were downloaded. Each of the data files contains information about
the weather stations, including the geographical position (latitude and 
longitude). These coordinates were plotted on a map (see Figure on the right),
and the closest station to the bulk of the data was chosen (station name: 
*MCTAVISH*). Only data from this station was used.



</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 43%"><!-- ---- start of second column              --> 
<img src="img/map-of-bike-and-possible-weather-stations_html.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: Plot shows all starting stations of a bike trip (red dots), as
well as the closest weather stations (blue markers). 
The closest station in the center is the *MCTAVISH* weather station 
(Climate Identifier 7024745, WMO Identifier 71612)
<p>

</div><!-- ------------------------------------ end of second column                -->




## Findings: Data Exploration

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 53%"><!-- ---- start of first column               -->

To get a better understanding of the data, the number of hourly bike trips
was visualized for the time span between $2014$ and $2017$. 

The moving average that is shown in the plot (red line) can 
be interpreted as a *baseline model*, i.e., 
the simplest possible model to describe the hourly number of bike rides. 

This baseline model explains $38.8\%$ of the variance $(r^2 = 0.388)$ and has
a mean absolute error of $MAE = 316.2$, which means that on average, the "prediction" 
for the number of hourly bike rides is wrong by this many bike rides. 
This includes also winter months with no rides. For a more realistic estimation
of model quality, these numbers drop to $r^2 = .079$ and $MAE = 510.7$ when only 
considering the time frame from May to September.

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 43%; margin-top: -3%"><!-- ---- start of second column              --> 
<img src="img/expl-trips-per-hour-2014-2017.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: Number of hourly rides from $2014$ to $2017$. Each dot represents the 
number of trips in one specifc hour. Red line represents a 
moving average using a window of $14$ days.
<p>

</div><!-- ------------------------------------ end of second column                -->


## Findings: Data Exploration

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 53%"><!-- ---- start of first column               -->

To visualize the relations between the available features,
a correlation heatmap is shown on the right. The features are only
slightly correlated, with the only exception being temperature and dew point
that show an almost perfect (linear) relationship $(r = .93)$.

To avoid problems resulting from this multicollinearity, 
only temperature was used as a predictor,
and dew point was dropped. Despite the fact that that gradient boosting
is less influenced by multicollinearity, 
it might still influence calculations of variable importance  [@rf_or_gbm; @rf_imp].

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 43%; margin-top: 0%"><!-- ---- start of second column              --> 
<img src="img/expl-corr-heatmap.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: Pearson Correlations between available features in the data.
<p>

</div><!-- ------------------------------------ end of second column                -->


## Findings: Model Fit

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 53%"><!-- ---- start of first column               -->

<p style="font-size: 12px">
**Table**: Model performance measures for different models. $r^2$ is the amount 
of variance explained, $MAE$ stands for mean absolute error. $train$ and $test$ 
specify training and testing set, respectively. For the test set, some performance
measures were also re-computed for using only the summer months of the test set 
(May to September), indicated via the $summer$ subscripts.
</p>

<p style="margin-top: -4%">
|Model                        |  $r^2_{train}$|   $r^2_{test}$| $r^2_{summer}$| $MAE_{test}$| $MAE_{summer}$|
|:----------------------------|--------------:|--------------:|-------------------:|------------:|-------------------:|
|Gradient Boosting (XGBoost)  |        $0.889$|        $0.860$|                  NA|      $158.0$|                  NA|
|Random Forest                |        $0.913$|        $0.894$|                  NA|      $111.2$|                  NA|
|Gradient Boosting (sklearn)  |        $0.997$|        $0.941$|             $0.933$|       $85.4$|             $105.4$|
</p>

The explained variance of the different models tried ranged from 
$r_{test}^2 = 0.860$ to $r_{test}^2 = 0.941$ for the final model 
(all in the test set; see table).
The hyperparameters for the final model (gradient boosting via scikit-learn) 
were selected via randomized search using $4$-fold cross validation 
(using only the training set). 

The final model fits the data very well, explaining $94.1\%$ of the variance and 
exhibiting an average error of $85.4$ rides over all hourly predictions. 
This error only increases to $105.4$ if only summer months are considered. 


</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 43%; margin-top: -3%"><!-- ---- start of second column              --> 
<img src="img/mod_gb_plot-pred-vs-true.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: The figure shows actual (true) values vs. predicted values for the
number of hourly bike trips for the final model. 
A perfect model would yield predictions that are
identical to the true values, i.e., all points would be on the $45°$ diagonal. 
This model is relatively close.
<p>

</div><!-- ------------------------------------ end of second column                -->





## Findings: Most Important Features

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 53%"><!-- ---- start of first column               -->

The most important features that influenced the prediciton of hourly bike 
rides were temperature and atmospheric pressure. While the former is easily 
comprehensible, the latter is best understood as a proxy for precipitation,
which was not available from the weather data: low pressure is commonly
related to rainy weather [@pressure_and_rain]. While Relative humidity is 
not as tightly connected to rain [@humidity_and_rain], it interacts with 
temperature, e.g., it influences how (high) temperatures are perceived 
[@humidity_and_temp]. 

Further important predictors are the hour of the day and the day of the week,
as well as wind direction and speed. How these features influence the 
predicted number of bike trips is best detailed by specific plots, so-called
*Partial Dependence Plots* (*PDP*) and *Individual Conditional Expectation* (*ICE*)
plots [@ice_plots].

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 43%; margin-top: 0%"><!-- ---- start of second column              --> 
<img src="img/mod_gb_plot-variable-importance.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: 
<p>

</div><!-- ------------------------------------ end of second column                -->


## Findings: Main effects

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 59%"><!-- ---- start of first column               -->

*Individual Conditional Expectation* (*ICE*) plots [@ice_plots] to quantify
the main effects of some of the predictors.

For details, see figure caption.

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 37%; margin-top: -15%"><!-- ---- start of second column              --> 
<img src="img/mod_gb_ice-main-standard---Qhr_of_day.jpg" width="100%" style="display: block; margin: auto auto auto 0;" /><img src="img/mod_gb_ice-main-standard---QTempC.jpg" width="100%" style="display: block; margin: auto auto auto 0;" /><img src="img/mod_gb_ice-main-standard---QRelHum.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />

</div><!-- ------------------------------------ end of second column                -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


<div style="float: left; width: 20%"><br></div>
<div style="float: left; width: 42%; margin-top: -10%">
<p style="font-size: 12px">
**Figure**: Figure Caption ... Figure Caption ... Figure Caption ... 
Figure Caption ... Figure Caption ... Figure Caption ... Figure Caption ... 
Figure Caption ... Figure Caption ... Figure Caption ... Figure Caption ... 
<p>
</div>
<div style="float: left; width: 38%"><br></div>
<div style="clear: both"></div>


## Findings: Interactions

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 53%"><!-- ---- start of first column               -->

*Individual Conditional Expectation* (*ICE*) plots [@ice_plots] to quantify
the main effects of some of the predictors.

For details, see figure caption.

</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 4%"><br></div><!-- spacing column ----------------- -->
<div style="float: left; width: 43%; margin-top: -10%"><!-- ---- start of second column              --> 
<img src="img/mod_gb_pdp-interact---Qhr_of_day--Qday_of_week.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
<p style="font-size: 12px">
**Figure**: 
<p>
</div><!-- ------------------------------------ end of second column                -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


## Findings: Interactions

<div></div><!-- ------------------------------- needed, but don't put anything here -->
<div style="float: left; width: 43%; margin-top: -10%"><!-- ---- start of first column               -->
<img src="img/mod_gb_pdp-interact---Qhr_of_day--QTempC.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />
</div><!-- ------------------------------------ end of first column                 -->
<div style="float: left; width: 14%"><br>
<!-- <div id="rot" style="margin-left: -10%; margin-right: -10%"> -->
<p style="font-size: 12px">
**Figure**: Figure ... Figure ... Figure ... Figure ... Figure ... Figure ... 
Figure ... Figure ... Figure ... Figure ... Figure ... Figure ... 
Figure ... Figure ... Figure ... Figure ... Figure ... Figure ... 
Figure ... Figure ... Figure ... Figure ... Figure ... Figure ... 
Figure ... Figure ... Figure ... Figure ... Figure ... Figure ... 
Figure ... Figure ... Figure ... Figure ... Figure ... Figure ... 
Figure ... Figure ... Figure ... Figure ... Figure ... Figure ... 
<p>
<!-- </div> -->
</div><!-- spacing column ----------------- -->
<div style="float: left; width: 43%; margin-top: -10%"><!-- ---- start of second column              --> 
<img src="img/mod_gb_pdp-interact---QRelHum--QTempC.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />

</div><!-- ------------------------------------ end of second column                -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


## Findings

Findings Outline:

* Data Exploration: Mean as baseline model?
* Correlation map?
* Model and Model Quality (+ pic predicted vs. actual), MAE, r^2 
    * numerical features only, also categorical models tried (but not as good)
* Important variables
* PDP for important variables
* Interaction Plots for important variables
    * most important interactions picked by rerunning the model (without CV) with only interaction terms

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
 
## References {.columns-2 .tiny}


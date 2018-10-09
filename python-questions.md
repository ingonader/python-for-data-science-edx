# Python-Questions

> Collected during  EdX Course "Python for Data Science"



* How to run / execute an analysis with multiple source files? In Juypterlab?
  * Which command to use for running file? How to execute a python script (similar to R's `source()`, with magic commands `%` in the script?)
  * Easy way to execute different files in the same console?
* JupyterLab

  * syntax completion in editor of jupyterlab?
  * any other good alternative as IDE? free?
  * Can RStudio be used for Python?
  * What about Spyder, anyone using this still, nowadays?
* Data Preparation (low-level)
  * Why are so many things of data type `object`? What does this mean? [example](./week-06-mini-project/imdb-movie-dataset-analysis.py-200-## turns 'complexity' into type 'object' again...)

  * How does indexing work in pandas? What is a multi-index, and how do I work with it? Is it useful or a pain in the ass? [example](./week-06-mini-project/imdb-movie-dataset-analysis.py:146:##GD this does not only rename columns, but replaces the multiindex with a flat one - try)

  * Time-Index? [example](./week-09-and-10-final-project/kgl-cycle-share-04-data-prep.py-40-    pd.DatetimeIndex(dat_trip_raw['start_date']), )

  * Adding an Index in-place vs. not? [example](./week-09-and-10-final-project/kgl-cycle-share-04-data-prep.py-39-dat_trip_raw.set_index)

  * Variable names that don't conform to standards -- how much of a problem is that in Python / pandas / scikit-learn ?

  * When to copy dataframes, when to modify in place? [example](./week-07-machine-learning/Weather Data Classification using Decision Trees.ipynb-1234-    "## note: course instructurs copy the data frequently\n",)

  * How to work with copies when modifying variables or creating new ones in a dataframe? (doesn't work like a view in SQL, does it? Add new stuff to original data and automatically see it in the copy, right?)

  * What does this warning mean ([example](./week-06-mini-project/imdb-movie-dataset-analysis.py-541-## A value is trying to be set on a copy of a slice from a DataFrame.
    .)):

    ```python
    # dat_test_y = (dat_test_y - ztrans_mean_y) / ztrans_sd_y
    ## produces warning:
    ## /Users/ingonader/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3137: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable## /indexing.html#indexing-view-versus-copy
    ##   self[k1] = value[k2]
    ```

* Missing Values:
  * How do missing values work ...
    * in Python?
    * in Numpy?
    * in Pandas?
  * is there also a `np.None`, on top of `np.nan`? [example](./week-06-mini-project/imdb-movie-dataset-analysis.py:217:##GD take care, None is not equivalent to np.nan (.astype(float) above converts None to np.nan)
  * is there a similar distinction between `NaN` and `None` as in R?
  * What is a good package to impute missing values?

* Plotting
  * What common plotting libraries are there in Python?
  * Are there different plotting systems in Python, similar to R's base, lattice/grid, and ggplot plotting system?
  * How to do map plots (with city street information) in Python?
  * How to modify (matplotlib) plots that result from different functions? (e.g., PDP plots?) [example](./week-09-and-10-final-project/kgl-cycle-share-06a-random-forest.py:213:## [[here]] [[?]] how to set axis labels?)
* Data Prep (high-level) and Modeling:
  * How to best normalize data in a pipeline? 
  * `StandardScaler` seems to lose variable names? [example](./week-06-mini-project/imdb-movie-dataset-analysis.py-568-# dat_train_x = scaler.transform(dat_train_x))
  * How do you (Günther) do modeling (train/val/test split, model formula)?
  * how to do dummy coding? `pd.get_dummies`, or use `patsy`? [example](./week-09-and-10-final-project/kgl-cycle-share-06a-random-forest.py-93-dat_y, dat_x = patsy.dmatrices(formula_txt, dat_hr_all, ))
  * How to best store models?
    * pickle or joblib? -- Limitations! <https://pyvideo.org/pycon-us-2014/pickles-are-for-delis-not-software.html> or <http://scikit-learn.org/stable/modules/model_persistence.html>
    * Other methods?
  * Are all scikit-learn models comparable or similar in their APIs? 
  * Random Forest: How to plot OOB error, like in R? [example](./week-09-and-10-final-project/kgl-cycle-share-06a-random-forest.py:137:## [[?]] missing: how to plot oob error by number of trees, like in R?)


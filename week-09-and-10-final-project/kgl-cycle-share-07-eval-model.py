## ######################################################################### ##
## Analysis of 
## For EdX Course
## Python for Data Science (Week 9 and 10 Final Project)
## ######################################################################### ##

## ========================================================================= ## 
## import libraries
## ========================================================================= ##

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## ========================================================================= ##
## load model from disk
## ========================================================================= ##

from sklearn.externals import joblib

## select file and define prefix (for plot output files):
#filename_model = 'model_random_forest.pkl'; filename_out_prefix = 'mod_rf_'; n_jobs = -2
#filename_model = 'model_random_forest_interact.pkl'; filename_out_prefix = 'mod_rfx_'; n_jobs = -2
filename_model = 'model_gradient_boosting.pkl'; filename_out_prefix = 'mod_gb_'; n_jobs = 1
#filename_model = 'model_xgb.pkl'; filename_out_prefix = 'mod_xgb_'; n_jobs = 1
#filename_model = 'model_nonzero_gradient_boosting.pkl'; filename_out_prefix = 'mod_nz_gb_'; n_jobs = 1


## load model:
mod_this = joblib.load(os.path.join(path_out, filename_model))

## define number of grid points for pdp interaction plots:
num_grid_points = [20, 20]

## ========================================================================= ##
## make predictions and get model performance
## ========================================================================= ##

## Make predictions using the testing set
dat_test_pred = mod_this.predict(dat_test_x)
dat_train_pred = mod_this.predict(dat_train_x)

## Inspect model:
mean_squared_error(dat_train_y, dat_train_pred)  # MSE in training set
mean_squared_error(dat_test_y, dat_test_pred)    # MSE in test set
mean_absolute_error(dat_train_y, dat_train_pred) # MAE in training set
mean_absolute_error(dat_test_y, dat_test_pred)   # MAE in test set
r2_score(dat_train_y, dat_train_pred)            # R^2 (r squared) in test set
r2_score(dat_test_y, dat_test_pred)              # R^2 (r squared) in test set

## r2 for non-zero counts:
r2_score(dat_test_y[dat_test_y > 0], dat_test_pred[dat_test_y > 0])

## ------------------------------------------------------------------------- ##
## variable importance
## ------------------------------------------------------------------------- ##

## variable importance:
var_imp = pd.DataFrame(
    {'varname_q'   : dat_train_x.columns,   ## quoted
     'varname_orig'   : [i[3:-2] for i in dat_train_x.columns], 
    'importance' : list(mod_this.feature_importances_)})
dat_varnames_long = pd.DataFrame.from_dict(varnames_long_dict, orient = 'index', columns = ['varname'])
var_imp = pd.merge(var_imp, dat_varnames_long, left_on = 'varname_q', right_index = True)
var_imp.sort_values('importance')#['varname']

## sort variables by importance for plotting:
varname_list = list(var_imp.sort_values('importance')['varname'])
varname_cat = CategoricalDtype(categories = varname_list, ordered=True)
var_imp['varname_cat'] = \
    var_imp['varname'].astype(str).astype(varname_cat)

## plot variable importance (15 most important):
p = ggplot(var_imp[-15:], aes(y = 'importance', x = 'varname_cat')) + \
    geom_bar(stat = 'identity') + \
    labs(
        title = "Feature importance",
        x = "Feature",
        y = "Importance") + \
    coord_flip()
print(p)

filename_this = 'plot-variable-importance.jpg'
#filename_this = 'plot-variable-importance-with-interactions.jpg'
ggsave(plot = p, 
       filename = os.path.join(path_out, filename_out_prefix + filename_this),
       height = 6, width = 6, unit = 'in', dpi = 300)

## ------------------------------------------------------------------------- ##
## partial dependence plots: main effects
## ------------------------------------------------------------------------- ##

from pdpbox import pdp, get_dataset, info_plots

# Package scikit-learn (PDP via function plot_partial_dependence() ) 
#   http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html
# Package PDPbox (ICE, c-ICE for single and multiple predictors) 
#   https://github.com/SauceCat/PDPbox 

#pd.merge(dat_train_x, pd.DataFrame(dat_train_y), left_index = True, right_index = True)
#dat_train_x.join(dat_train_y)  ## identical


#dat_train_x.columns

%matplotlib inline


plot_params_default = {
            # plot title and subtitle
            #'title': 'Partial Dependence for: %s' % varnames_long_dict[wch_feature],
            'subtitle': '',
            'title_fontsize': 20,
            'subtitle_fontsize': 12,
            'font_family': 'Arial',
            # matplotlib color map for ICE lines
            'line_cmap': 'Blues',
            'xticks_rotation': 0,
            # pdp line color, highlight color and line width
            'pdp_color': '#1A4E5D',
            'pdp_hl_color': '#FEDC00',
            'pdp_linewidth': 1.5,
            # horizon zero line color and with
            'zero_color': '#E75438',
            'zero_linewidth': 1,
            # pdp std fill color and alpha
            'fill_color': '#66C2D7',
            'fill_alpha': 0.2,
            # marker size for pdp line
            'markersize': 3.5,
        }

## pdp (and then ice plot) calculation for numeric feature:
#features[1]
wch_feature = "Q('Temp (°C)')"
plot_params_default.update({
        'title': 'Partial Dependence for: %s' % \
        varnames_long_dict[wch_feature]
    })
pdp_current = pdp.pdp_isolate(
    model = mod_this, dataset = dat_train_x.join(dat_train_y), 
    num_grid_points = 20, n_jobs = n_jobs, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, 
    feature = wch_feature
)

## pdp plot for numeric features:
#%matplotlib osx
fig, axes = pdp.pdp_plot(
    pdp_current, varnames_long_dict[wch_feature], #wch_feature, 
    plot_params = plot_params_default
)
filename_this = "pdp-main---" + pv.sanitize_python_var_name(wch_feature) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_out_prefix + filename_this), dpi = 300)

## ice-plot for numeric feature:
fig, axes = pdp.pdp_plot(
    pdp_current, wch_feature, plot_lines = True, frac_to_plot = 100,  ## percentage! 
    x_quantile = False, plot_pts_dist = True, show_percentile = True)
filename_this = "ice-main---" + pv.sanitize_python_var_name(wch_feature) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_out_prefix + filename_this), dpi = 300)

## [[here]] [[?]] how to set axis labels?



## ------------------------------------------------------------------------- ##
## partial dependence plots: interactions
## ------------------------------------------------------------------------- ##

#[features[6], features[5]]
wch_features = ["Q('hr_of_day')", "Q('Stn Press (kPa)')"]
inter_current = pdp.pdp_interact(
    model = mod_this, dataset = dat_train_x.join(dat_train_y),
    num_grid_points = num_grid_points, n_jobs = n_jobs, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, features = wch_features)
fig, axes = pdp.pdp_interact_plot(
    inter_current, wch_features, x_quantile = False, 
    plot_type = 'contour', plot_pdp = False
)
filename_this = "pdp-interact---" + \
    pv.sanitize_python_var_name(wch_features[0]) + "--" + \
    pv.sanitize_python_var_name(wch_features[1]) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_out_prefix + filename_this), dpi = 300)

#[features[6], features[7]]
wch_features = ["Q('hr_of_day')", "Q('day_of_week')"]
inter_current = pdp.pdp_interact(
    model = mod_this, dataset = dat_train_x.join(dat_train_y),
    num_grid_points = num_grid_points, n_jobs = n_jobs, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, features = wch_features)
fig, axes = pdp.pdp_interact_plot(
    inter_current, wch_features, x_quantile = False, 
    plot_type = 'contour', plot_pdp = False
)
filename_this = "pdp-interact---" + \
    pv.sanitize_python_var_name(wch_features[0]) + "--" + \
    pv.sanitize_python_var_name(wch_features[1]) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_out_prefix + filename_this), dpi = 300)

#[features[1], features[2]]
wch_features = ["Q('Temp (°C)')", "Q('Rel Hum (%)')"]
inter_current = pdp.pdp_interact(
    model = mod_this, dataset = dat_train_x.join(dat_train_y),
    num_grid_points = num_grid_points, n_jobs = n_jobs, ## needs to be 1 for XGBoost model!
    model_features = dat_train_x.columns, features = wch_features)
fig, axes = pdp.pdp_interact_plot(
    inter_current, wch_features, x_quantile = False, 
    plot_type = 'contour', plot_pdp = False
)
filename_this = "pdp-interact---" + \
    pv.sanitize_python_var_name(wch_features[0]) + "--" + \
    pv.sanitize_python_var_name(wch_features[1]) + ".jpg"
fig.savefig(fname = os.path.join(path_out, filename_out_prefix + filename_this), dpi = 300)

## ------------------------------------------------------------------------- ##
## other plots
## ------------------------------------------------------------------------- ##


## target distribution for numeric feature:
wch_feature = features[1]
fig, axes, summary_df = info_plots.target_plot(
    df = dat_train_x.join(dat_train_y),
    feature = wch_feature,
    feature_name = wch_feature, target = target, 
    show_percentile = True
)

# ## check prediction distribution for numeric feature
# ## (doesn't work?)
# wch_feature = features[1]
# fig, axes, summary_df = info_plots.actual_plot(
#     model = mod_this, X = dat_train_x, 
#     feature = wch_feature, feature_name = wch_feature, 
#     show_percentile = True
# )

# ## visualize a single tree:
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = mod_this.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = dat_x.columns, 
#                 rounded = True, precision = 1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')



## ========================================================================= ##
## plot data with predictions
## ========================================================================= ##

## make predictions for complete dataset:
dat_y['pred'] = mod_this.predict(dat_x)
dat_y.head()

## add to original dataset:
dat_hr_all = pd.merge(dat_hr_all, 
                      dat_y[['pred']], 
                      how = 'left',
                      left_index = True,
                      right_index = True)

## plot predictions vs. real value of target:
p = ggplot(dat_y, aes(x = "Q('trip_cnt')", y = 'pred')) + geom_point(alpha = .1)
print(p)
filename_this = 'plot-pred-vs-true.jpg'
ggsave(plot = p, 
       filename = os.path.join(path_out, filename_out_prefix + filename_this),
       height = 6, width = 6, unit = 'in', dpi = 300)

## line plot of number of trips per hour:
p = ggplot(dat_hr_all, aes(y = 'trip_cnt', x = 'start_date')) + \
    geom_point(alpha = .05, color = 'black') + \
    geom_point(aes(y = 'pred'), alpha = .05, color = 'orange') + \
    geom_smooth(method = 'mavg', method_args = {'window' : 14*24}, 
                color = 'red', se = False)
print(p)
## not worth saving -- doesn't show anything of interest.
## (except maybe some missing values in predictions in the spring of 2014... why?)




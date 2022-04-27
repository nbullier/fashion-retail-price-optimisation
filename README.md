# Price optimisation model

This repo contains the S2DS implementation of fashion retail price optimisation model.

# Directory structure: 

```
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   └── raw            <- The original data.
│
├── notebooks          <- Jupyter notebooks of various models.
│
├── references         <- Paper references on price optimisation.
│
├── src                <- Complete model pipeline for this project.

```

# Notebook descriptions

- 1.0-data_cleaning_selection.ipynb * Notebook contains code for preparing and cleaning data for next steps.
- 1.1-data_synthesis_manual.ipynb * Notebook contains steps for resampling original data and visualisation of resampled data.
- 1.4-aa-append_synth_engin_cols.ipynb * Notebook contains steps for data preparation, including aggregating data by week, generating engineered columns,              appending synthetic star rating and sentiment columns, and adding google trend data for items in the subdepartment of interest and the colour of items considered.
- 1.5-aa_decision_tree.ipynb Notebook contains an initial exploration of the use of decision trees to predict demand.
- 1.6_et_linear_regression.ipynb Notebook contains use of and assessment of linear regression models for demand prediction
- 1.6-aa-random_forest.ipynb Notebook contains an initial exploration of the use of random forests to predict demand. Final selection of the best random forest model is carried out in ‘2.0_et_optimised_random_forest.ipynb’.
- 1.6-nb-optimisation-lp-mip.ipynb * LP/MIP solver using the methodology described in **Analytics for an Online Retailer: Demand
Forecasting and Price Optimization** [link to the paper](https://pubsonline.informs.org/doi/10.1287/msom.2015.0561)
- 1.7_et_randomsearchCV_v_rf_comparison.ipynb Notebook contains workup of splitting training data for cross validation and comparison of random forest manually optimised verses hyperparameter tuned.
- 1.7-aa-predict_demand_optimise_price.ipynb * Notebook brings together the POC model, including the demand prediction and price optimisation step. Currently, the step that builds the demand matrix is slow and therefore we do not recommend searching a price space that is too large. 
- 2.0_et_optimised_random_forest.ipynb * Notebook contains steps getting to final optimised random forest model for demand prediction.



# Running the complete pipeline
The star * indicates which notebooks are directly used in the final pipeline file in the src folder: price_optim.py



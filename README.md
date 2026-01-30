# Walmart Weekly Sales Forecasting Using XGBoost
## Abstract

This study presents a machine learning–based framework for forecasting weekly sales across Walmart stores using historical sales and exogenous economic variables.

A time-series–aware modeling pipeline is developed that preserves temporal ordering, mitigates data leakage, and captures trend and seasonality through structured feature engineering.

An XGBoost regression model is employed to generate 12-week ahead recursive forecasts for each store, accompanied by 95% confidence intervals to quantify predictive uncertainty.

## Introduction

Accurate demand forecasting is a critical component of retail operations, influencing inventory management, pricing strategies, and supply chain optimization.

Traditional statistical time-series models often struggle with non-linear patterns and high-dimensional covariates.

This project explores the application of gradient-boosted decision trees to time-series sales forecasting through explicit temporal feature construction.

## Dataset Description

The dataset consists of weekly observations for multiple Walmart stores, containing:

- Store identifier

- Date

- Weekly sales (target variable)

- Temperature

- Fuel price

- Consumer Price Index (CPI)

- Unemployment rate

Each store is modeled as an independent univariate time series with exogenous regressors.

## Why XGBoost Regression?

- Effectively captures non-linear relationships in retail sales data

- Performs exceptionally well on tabular, feature-engineered time-series data

- Leverages lag, rolling, and calendar features for supervised forecasting

- Robust to noise and outliers caused by holidays and promotions

- Built-in regularization helps prevent overfitting in recursive forecasts

- Scales efficiently across multiple stores and long time horizons

- Proven strong performance in retail forecasting and Kaggle competitions

- Integrates seamlessly with recursive multi-step forecasting strategies

## Methodology

### Data Preprocessing

Dates are converted to a datetime format and sorted chronologically.

The target variable (Weekly_Sales) is transformed using a logarithmic function (log1p) to stabilize variance and reduce heteroscedasticity.

### Feature Engineering

To encode temporal dependencies, the following features are constructed:

**Calendar-Based Features**

Extracted from the date index:

- Year

- Month

- Week of year

- Day of week

These features enable the model to learn seasonal and cyclical effects.

**Lag Features**

Historical sales values are introduced as predictors:

Short-term lags (e.g., 1–2 weeks)

Seasonal lag (lag_52) capturing year-over-year effects

**Rolling Statistics**

Rolling means of past sales:

4-week rolling average

8-week rolling average

All lagged and rolling features are shifted forward to ensure strict causality.

### Train–Test Splitting

A time-based split is applied:

- Training set: observations prior to 2012

- Test set: observations from 2012 onward

This approach avoids look-ahead bias and reflects real-world forecasting conditions.

## Model Specification

### XGBoost Regressor

An XGBoost regression model is trained using the engineered features. Key hyperparameters include:

- Number of estimators: 500

- Maximum tree depth: 6

- Learning rate: 0.05

- Subsampling and column sampling for regularization

XGBoost is selected due to its ability to:

- Model complex non-linear relationships

- Handle high-dimensional tabular data

- Incorporate regularization to mitigate overfitting

## Evaluation

### Performance Metric

Model performance is evaluated using Mean Absolute Percentage Error (MAPE), which provides an interpretable percentage-based error measure commonly used in sales forecasting.

## Forecasting Strategy

### Recursive Multi-Step Forecasting

A recursive forecasting approach is employed to generate 12-week ahead predictions:

- The model predicts sales for week 𝑡 + 1

- The prediction is appended to the feature set

- The process is repeated iteratively until the forecast horizon is reached

This approach reflects practical forecasting scenarios where future observations are unavailable.

## Uncertainty Quantification

Prediction uncertainty is estimated using empirical residuals from the test set.

Assuming approximately normal residuals, 95% confidence intervals are computed as:

$\hat{y} \pm 1.96 \cdot \sigma$

where $\sigma$ denotes the standard deviation of the residuals.

## Results

The model produces:

- Store-level 12-week sales forecasts

- Aggregated forecasts across all stores

- Upper and lower confidence bounds for each prediction

Results demonstrate the effectiveness of tree-based models when combined with explicit temporal feature engineering for retail demand forecasting.

## Reproducibility

To reproduce the results:

- Clone the repository

- Open walmart_weekly_sales_forecast.ipynb

- Install required dependencies:

    pandas

    numpy

    matplotlib

    scikit-learn

    xgboost

- Execute all notebook cells sequentially

## Potential Extensions

-  Incorporation of holiday and promotion indicators

- Store-specific hyperparameter optimization

- Comparison with classical (ARIMA, SARIMA) and deep learning (LSTM) models

- Probabilistic forecasting using quantile regression


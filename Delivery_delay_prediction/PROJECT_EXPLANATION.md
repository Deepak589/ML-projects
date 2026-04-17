# Delivery Time Prediction Project

## Short Project Summary

This project predicts delivery time for customer orders using historical delivery data. It uses two complementary machine learning models:

- A regression model that estimates delivery time in minutes.
- A classification model that predicts whether an order is likely to take longer than the dataset's typical delivery duration.

The final model is deployed through a Streamlit app where users can enter order, route, agent, traffic, weather, and category details to receive an estimated delivery time and an above-typical risk score.

## Problem Statement

Delivery platforms need reliable ETA estimates so customers, stores, and operations teams can plan better. The goal of this project is to build a machine learning system that estimates delivery duration and identifies orders that may take longer than normal based on historical patterns.

The classification target is not a strict business SLA delay label. Instead, it is defined using the median delivery time from the dataset. Orders above that threshold are treated as "above typical." This keeps the classification problem balanced and avoids labeling nearly every order as delayed.

## Dataset and Features

The project uses an Amazon delivery dataset with order, agent, location, weather, traffic, vehicle, area, delivery time, and category information.

Important engineered features include:

- `distance_km`: calculated using the Haversine formula from store and drop coordinates.
- `order_hour`: extracted from order time.
- `is_weekend`: created from the order timestamp.
- `prep_time`: time between order placement and pickup, presented in the app as order processing time.
- one-hot encoded categorical features for weather, traffic, vehicle, area, category, and order day.

Data quality was improved by removing unrealistic route distances above `50 km`, because those rows likely came from bad coordinates and created extreme prediction errors.

## Modeling Approach

The project uses Random Forest models:

- `RandomForestRegressor` for delivery-time prediction.
- `RandomForestClassifier` for above-typical risk prediction.

Hyperparameter tuning was performed with Optuna. Optuna was chosen because it explores the hyperparameter space more efficiently than manual tuning or exhaustive grid search.

The workflow includes:

- data cleaning
- feature engineering
- train/test split
- baseline model training
- Optuna hyperparameter tuning
- tuned model evaluation
- model saving with `joblib`
- Streamlit deployment

## Current Model Performance

Regression model:

- MAE: about `16.55 minutes`
- RMSE: about `21.18 minutes`
- R2: about `0.82`

Classification model:

- Accuracy: about `0.846`
- F1-weighted: about `0.845`
- ROC-AUC: about `0.926`

After distance cleanup, the number of extreme `100+ minute` regression errors dropped significantly. The model is useful for estimating typical delivery behavior, while rare edge cases can still produce larger misses.

## How the Two Models Are Used

The two models do not depend on each other. They solve different but related tasks.

The regression model is the main ETA estimator:

- Example output: `Estimated delivery time: 118 minutes`

The classification model is a supporting risk signal:

- Example output: `Above-typical probability: 34%`

This dual-output design is stronger than forcing the models into a staged pipeline because the dataset did not show that a second correction model improved performance.

## Streamlit App

The Streamlit app allows a user to enter:

- agent age and rating
- order processing time
- order date and time
- store and drop coordinates
- weather
- traffic
- vehicle type
- area
- order category

The app returns:

- estimated delivery time
- estimated ETA
- above-typical probability
- risk level
- probability breakdown
- model performance summary

The app also warns users when route distance is outside the cleaned training range.

## Resume Bullet Points

- Built an end-to-end delivery time prediction system using Python, pandas, scikit-learn, Optuna, and Streamlit.
- Engineered route, time, and categorical features including Haversine distance, order hour, weekend flag, and order processing time.
- Trained Random Forest regression and classification models to estimate delivery duration and identify above-typical delivery risk.
- Improved data quality by filtering unrealistic route distances, reducing extreme `100+ minute` prediction errors.
- Tuned model hyperparameters with Optuna and achieved approximately `0.82 R2` for ETA prediction and `0.926 ROC-AUC` for above-typical risk classification.
- Deployed the trained models in an interactive Streamlit app with user input forms, risk scoring, ETA output, and model-performance context.

## Interview Explanation

I built a delivery time prediction project using historical order data. The main model is a Random Forest regressor that predicts delivery time in minutes. I also trained a separate Random Forest classifier that estimates whether an order is likely to take longer than the dataset's typical delivery duration.

One important decision was how to define "delay." A fixed threshold like 30 minutes would have made almost every order delayed in this dataset, so I used the median delivery time as a balanced above-typical threshold. This made the classification task more meaningful and learnable.

I engineered features such as Haversine distance, order hour, weekend flag, and order processing time. I also cleaned unrealistic route distances above 50 km because those were likely coordinate errors and were causing rare but very large prediction mistakes.

For tuning, I used Optuna because it is more efficient than exhaustive grid search. The final regression model achieved around 0.82 R2, and the classifier achieved around 0.926 ROC-AUC. I deployed the models in Streamlit so users can enter delivery conditions and receive both an ETA and a risk score.

## Limitations and Future Improvements

- The classification label is statistical, not a business SLA label.
- Real-time traffic and weather APIs would make the app more realistic.
- The dataset has some coordinate-quality issues that need filtering.
- Some rare long-delivery cases are still difficult to predict.
- Future work could include gradient boosting models, SHAP explanations, batch CSV prediction, and live route-distance APIs.

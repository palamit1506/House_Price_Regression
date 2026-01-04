🏠 House Prices Prediction — Advanced Regression Techniques

Predicting residential house sale prices using advanced regression models and cross-validation on the Ames Housing dataset.

📌 Project Overview

House prices are influenced by far more than just square footage or the number of bedrooms.
This project builds a robust regression pipeline to predict the final sale price of houses by learning complex relationships from 79 structural, location-based, and quality-related features.

The goal is to develop models that generalize well and minimize RMSLE, the official Kaggle evaluation metric.

📂 Dataset
Attribute	Details
Source	Kaggle — House Prices: Advanced Regression Techniques
Training Samples	~1,460
Test Samples	~1,459
Features	79 explanatory variables
Target	SalePrice
Key Feature Groups

Property size & layout (GrLivArea, LotArea)

Quality & condition (OverallQual, OverallCond)

Location & neighborhood

Basement, garage, and exterior attributes

🧠 Methodology

🔹 Data Preprocessing

Handled missing values using feature-level semantics

Dropped identifier columns

Log-transformed target variable to reduce skewness

Encoded categorical variables

Optional feature scaling for linear models

🔹 Feature Engineering

Converted ordinal quality variables into numerical representations

Reduced skew in highly skewed numerical features

Ensured consistent preprocessing across train and test data

🔹 Validation Strategy

5-Fold Cross-Validation

Models retrained from scratch on each fold

Same folds used for all models to ensure fair comparison

🤖 Models Trained & Compared
Model	Notes
Linear Regression	Baseline
Gradient Boosting Regressor	Tree-based ensemble
LightGBM Regressor	Fast gradient boosting
CatBoost Regressor	⭐ Best generalization
🏆 Best Model Summary
Metric	Value
CV Metric	RMSLE
Best Model	CatBoost / LightGBM
Generalization	Strong & stable across folds
📊 Evaluation Metric

Root Mean Squared Log Error (RMSLE)

Penalizes relative prediction errors

Treats expensive and inexpensive houses equally

Matches Kaggle leaderboard evaluation

📉 Lower RMSLE indicates better model performance.

📁 Repository Structure
├── house-prices-regression.ipynb   # End-to-end regression pipeline
├── README.md

🧪 Test Set Predictions

Final predictions were generated using the best-performing model and inverse log transformation:

test_preds = np.expm1(best_model.predict(test_final))


Predictions were formatted according to Kaggle submission guidelines.

🚀 Key Learnings

Tree-based boosting models outperform linear baselines

Cross-validation provides a reliable estimate of real-world performance

Feature preprocessing significantly impacts regression accuracy

Gradient boosting effectively captures non-linear interactions

🛠 Tech Stack

Python

Pandas, NumPy

scikit-learn

LightGBM

CatBoost

Matplotlib / Seaborn

📚 Acknowledgments

The Ames Housing dataset was compiled by Dean De Cock for data science education and serves as a modern alternative to the Boston Housing dataset.

📌 This project demonstrates end-to-end regression modeling, cross-validation, and model comparison on a real-world dataset.

# Customer Churn Prediction 📊

A Machine Learning project that predicts whether a customer is likely to churn based on customer demographics, account information, and banking activity.

## 🚀 Live Demo

👉 [Try the Customer Churn Prediction App](https://customer-chrun-prediction.streamlit.app/)

## 📌 Project Overview

Customer churn refers to customers leaving or stopping their relationship with a company.

In this project, a Machine Learning classification model is used to predict whether a customer is likely to churn based on different customer and account-related features.

The trained model is integrated into an interactive **Streamlit web application**, where users can enter customer details and receive a churn prediction along with the predicted probability.

## 🎯 Objectives

- Analyze customer characteristics related to churn
- Perform data preprocessing
- Perform feature engineering
- Train a Machine Learning classification model
- Save the trained model using Joblib
- Build an interactive Streamlit application
- Deploy the Machine Learning model as a web application

## 📊 Features Used

The model uses the following customer information:

- Credit Score
- Country
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Ownership
- Active Membership
- Estimated Salary

## ⚙️ Feature Engineering

Additional features were created during the Machine Learning workflow:

- `balance_per_product` — Balance divided by number of products
- `salary_balance_ratio` — Estimated salary relative to account balance
- `age_group` — Customers grouped into different age categories
- `tenure_bucket` — Customers grouped based on tenure
- `high_balance` — Indicator for customers with a balance above 50,000

## 🤖 Machine Learning

The trained Machine Learning pipeline is saved as:

```text
churn_pipeline.pkl

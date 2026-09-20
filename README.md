# Customer Churn Prediction 📊

A machine learning project that predicts whether a customer is likely to churn based on customer demographics, account information, and banking activity.

## 🚀 Project Overview

Customer churn refers to customers leaving or stopping their relationship with a company.

In this project, a Machine Learning classification model is used to predict customer churn based on features such as:

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

The trained model is deployed as an interactive **Streamlit web application** where users can enter customer details and get a churn prediction along with the predicted churn probability.

## 🎯 Objectives

- Analyze customer characteristics related to churn
- Perform feature engineering
- Train a machine learning classification model
- Evaluate the model
- Save the trained model using Joblib
- Build an interactive Streamlit application
- Deploy the prediction system as a web app

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## ⚙️ Feature Engineering

Additional features were created to improve the model:

- `balance_per_product` — Balance divided by number of products
- `salary_balance_ratio` — Estimated salary relative to account balance
- `age_group` — Customers grouped by age
- `tenure_bucket` — Customers grouped by tenure
- `high_balance` — Indicator for customers with balance above 50,000

## 🤖 Machine Learning

The final trained pipeline is saved as:

```text
churn_pipeline.pkl

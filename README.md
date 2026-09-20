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
- Train and compare multiple Machine Learning classification models
- Identify and address class imbalance to improve churn detection
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

Five classification models were trained and compared using 5-fold cross-validation (ROC-AUC scoring):

| Model | CV ROC-AUC |
|---|---|
| Logistic Regression | 0.7877 |
| Random Forest | 0.8486 |
| **Gradient Boosting** | **0.8628** ⭐ Best |
| AdaBoost | 0.8462 |
| SVC | 0.8351 |

**Gradient Boosting** was selected as the best-performing model and evaluated on the held-out test set.

### Baseline Test Set Performance
| Metric | Score |
|---|---|
| Accuracy | 86.80% |
| Precision | 78.04% |
| Recall | 48.89% |
| F1-score | 60.12% |
| ROC-AUC | 86.92% |

### Feature Importance
Top predictors of churn: **customer age** and **number of products**, followed by `balance_per_product` and account activity status.

## 🔍 Model Improvement — Handling Class Imbalance

The dataset has a class imbalance (only ~20.4% of customers churned), which caused the baseline model to have strong accuracy but **low recall (48.9%)** — meaning it missed over half of the customers who actually churned. For a churn-prevention use case, this is a critical gap, since failing to flag an at-risk customer is costlier than a false alarm.

**Approach:** Applied SMOTE (Synthetic Minority Oversampling) combined with decision-threshold tuning to improve recall on the minority (churn) class.

### Before vs. After Comparison
| Metric | Baseline (Gradient Boosting) | Improved (SMOTE + Threshold = 0.4) | Change |
|---|---|---|---|
| Recall | 48.89% | **66.34%** | **+17.5 points** ✅ |
| F1-score | 60.12% | **63.75%** | +3.6 points ✅ |
| Precision | 78.04% | 61.36% | -16.7 points (trade-off) |
| Accuracy | 86.80% | ~86.30% | roughly stable |

This is a deliberate precision–recall trade-off: the improved model identifies significantly more at-risk customers, better supporting a real-world retention strategy, at the cost of some false positives.

The trained pipeline is saved as:

```text
churn_pipeline.pkl            # Baseline Gradient Boosting model
churn_pipeline_smote.pkl      # Improved model (SMOTE + threshold tuning) — used in the live app

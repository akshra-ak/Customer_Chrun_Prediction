import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("churn_pipeline_smote.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the probability of churn.")

st.divider()

# Customer inputs
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

country = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

products_number = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=10,
    value=2
)

credit_card = st.selectbox(
    "Has Credit Card?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

active_member = st.selectbox(
    "Active Member?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=60000.0
)


if st.button("🔮 Predict Churn"):

    # Create input dataframe
    sample_df = pd.DataFrame([{
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary
    }])

    # Same feature engineering used during model development

    sample_df["balance_per_product"] = (
        sample_df["balance"] /
        sample_df["products_number"].replace(0, np.nan)
    )

    sample_df["balance_per_product"] = (
        sample_df["balance_per_product"].fillna(0)
    )

    sample_df["salary_balance_ratio"] = (
        sample_df["estimated_salary"] /
        sample_df["balance"].replace(0, np.nan)
    )

    sample_df["salary_balance_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # For a single prediction, use 0 when balance = 0
    sample_df["salary_balance_ratio"].fillna(0, inplace=True)

    # Age group
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

    sample_df["age_group"] = pd.cut(
        sample_df["age"],
        bins=bins,
        labels=labels
    )

    # Tenure bucket
    sample_df["tenure_bucket"] = pd.cut(
        sample_df["tenure"],
        bins=[-1, 0, 2, 5, 10, 100],
        labels=["0", "1-2", "3-5", "6-10", "10+"]
    )

    # High balance
    sample_df["high_balance"] = (
        sample_df["balance"] > 50000.0
    ).astype(int)

    # Remove customer ID if present
    sample_df = sample_df.drop(
        columns=["customer_id"],
        errors="ignore"
    )

    # Prediction
    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

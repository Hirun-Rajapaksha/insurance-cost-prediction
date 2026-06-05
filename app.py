import pandas as pd
import numpy as np
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰",
    layout="centered"
)


# ---------------------------------------------------------
# Load cleaned dataset and train model
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    # Load cleaned dataset
    df = pd.read_csv("data/insurance_cleaned.csv")

    # Separate features and target
    X = df.drop(columns=["charges"])
    y = df["charges"]

    # Train Gradient Boosting model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    return model, X.columns


model, feature_columns = train_model()


# ---------------------------------------------------------
# App title
# ---------------------------------------------------------
st.title("Insurance Cost Prediction App")

st.write(
    """
    This web app predicts estimated medical insurance charges based on customer
    demographic and lifestyle information.
    """
)


# ---------------------------------------------------------
# User inputs
# ---------------------------------------------------------
st.header("Enter Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)


# ---------------------------------------------------------
# Feature engineering for input
# ---------------------------------------------------------
def prepare_input(age, sex, bmi, children, smoker, region, feature_columns):
    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "smoker_obese": 1 if smoker == "yes" and bmi >= 30 else 0,

        "sex_male": 1 if sex == "male" else 0,

        "smoker_yes": 1 if smoker == "yes" else 0,

        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,

        "bmi_category_normal": 1 if 18.5 <= bmi < 25 else 0,
        "bmi_category_overweight": 1 if 25 <= bmi < 30 else 0,
        "bmi_category_obese": 1 if bmi >= 30 else 0,

        "age_group_middle": 1 if 30 <= age < 50 else 0,
        "age_group_senior": 1 if age >= 50 else 0,
    }

    input_df = pd.DataFrame([input_data])

    # Ensure same column order as training data
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    return input_df


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("Predict Insurance Cost"):
    input_df = prepare_input(
        age,
        sex,
        bmi,
        children,
        smoker,
        region,
        feature_columns
    )

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Insurance Cost: ${prediction:,.2f}")

    st.write("### Input Summary")
    st.write(
        pd.DataFrame({
            "Feature": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
            "Value": [age, sex, bmi, children, smoker, region]
        })
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.write("Built using Python, Streamlit, Pandas, and Scikit-learn.")
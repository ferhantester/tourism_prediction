
import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained model committed to GitHub
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_model.joblib"
)

model_bundle = joblib.load(model_path)

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]
category_mappings = model_bundle["category_mappings"]
numeric_medians = model_bundle["numeric_medians"]


st.title("Wellness Tourism Package Prediction")

st.write("""
This application predicts whether a customer is likely to purchase
the Wellness Tourism Package based on customer and interaction details.
""")


# Customer inputs

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Unmarried", "Single", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=20,
    value=2
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

Designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    max_value=1000000.0,
    value=25000.0
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=60,
    value=10
)


# Save user inputs into a DataFrame

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])


# Make prediction

if st.button("Predict Package Purchase"):

    # Apply exactly the same categorical mappings used during training
    for col, mapping in category_mappings.items():
        input_data[col] = (
            input_data[col]
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    # Apply the same numeric medians used during training
    for col, median_value in numeric_medians.items():
        input_data[col] = input_data[col].fillna(median_value)

    # Ensure the columns are in exactly the training order
    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The model predicts that the customer is likely to purchase "
            "the Wellness Tourism Package."
        )
    else:
        st.info(
            "The model predicts that the customer is unlikely to purchase "
            "the Wellness Tourism Package."
        )

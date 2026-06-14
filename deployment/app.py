import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os
from pathlib import Path

APP_NAME = "Travel Wellness Lead Scoring"
MODEL_REPO_ID = os.getenv("TRAVEL_WELLNESS_MODEL_REPO_ID", "your-huggingface-username/travel-wellness-model")
MODEL_FILENAME = os.getenv("TRAVEL_WELLNESS_MODEL_FILENAME", "travel_wellness_model.joblib")
LOCAL_MODEL_PATH = Path(os.getenv("TRAVEL_WELLNESS_LOCAL_MODEL_PATH", Path(__file__).with_name(MODEL_FILENAME)))

@st.cache_resource
def load_model():
    if LOCAL_MODEL_PATH.exists():
        return joblib.load(LOCAL_MODEL_PATH)

    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(model_path)

st.set_page_config(page_title=APP_NAME, page_icon="", layout="centered")
st.title(APP_NAME)
st.write("Estimate whether a customer is likely to accept a wellness travel offer using demographic, travel, and sales-interaction details.")

try:
    model = load_model()
except Exception as exc:
    st.error(
        "Model could not be loaded. Set TRAVEL_WELLNESS_MODEL_REPO_ID and TRAVEL_WELLNESS_MODEL_FILENAME "
        "to your Hugging Face model repository and artifact name."
    )
    st.exception(exc)
    st.stop()

st.write("Enter the customer information below.")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business","Student", "Retired", "Other"]
)
designation = st.text_input("Designation", value="")

city_tier = st.selectbox("City Tier", [1, 2, 3])
monthly_income = st.number_input(
    "Monthly Income (Gross)", min_value=0, step=1000, value=50000
)
passport = st.selectbox("Valid Passport?", ["Yes", "No"])
own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
number_of_persons = st.number_input(
    "Number of People Visiting", min_value=1, max_value=10, value=2
)
number_of_children = st.number_input(
    "Number of Children Visiting", min_value=0, max_value=5, value=0
)

type_of_contact = st.selectbox(
    "Type of Contact", ["Company Invited", "Self Inquiry"]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King", "Other"]
)
pitch_satisfaction = st.slider(
    "Pitch Satisfaction Score", min_value=0, max_value=10, value=7
)
number_of_followups = st.number_input(
    "Number of Follow-ups", min_value=0, max_value=20, value=1
)
duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)", min_value=1, max_value=120, value=10
)
number_of_trips = st.number_input(
    "Average Number of Trips per Year", min_value=1, max_value=50, value=2
)
preferred_property_star = st.number_input(
    "Preferred Hotel Star Rating", min_value=1, max_value=5, value=3
)


input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "Occupation": occupation,
    "Designation": designation,
    "CityTier": city_tier,
    "MonthlyIncome": monthly_income,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfPersonVisiting": number_of_persons,
    "NumberOfChildrenVisiting": number_of_children,
    "TypeofContact": type_of_contact,
    "ProductPitched": product_pitched,
    "PitchSatisfactionScore": pitch_satisfaction,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfTrips": number_of_trips,
    "PreferredPropertyStar": preferred_property_star
}])


# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "accept the offer" if prediction == 1 else "decline the offer"
    st.write(f"Based on the information provided, the customer is likely to {result}.")

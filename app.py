import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

# ---------------------------------------------------
# Load and prepare the dataset
# ---------------------------------------------------
df = pd.read_csv("Cotton_Crop_Stages_Diseases.csv")

df["End_Day"] = df["Days from Sowing (Start)"] + df["Stage Duration (days)"]

le_stage = LabelEncoder()
le_disease = LabelEncoder()

df["Stage_encoded"] = le_stage.fit_transform(df["Crop Stage"])
df["Disease_encoded"] = le_disease.fit_transform(df["Crop Disease"])

X = df[["Days from Sowing (Start)", "Stage Duration (days)", "Stage_encoded"]]
y = df["Disease_encoded"]

model = DecisionTreeClassifier()
model.fit(X, y)


# ---------------------------------------------------
# Model Functions
# ---------------------------------------------------
def predict_stage(days_after_sowing):
    row = df[
        (df["Days from Sowing (Start)"] <= days_after_sowing) &
        (df["End_Day"] >= days_after_sowing)
    ]
    if not row.empty:
        return row["Crop Stage"].iloc[0]

    df["distance"] = abs(df["Days from Sowing (Start)"] - days_after_sowing)
    return df.loc[df["distance"].idxmin()]["Crop Stage"]


def predict_disease(stage_name):
    if stage_name not in le_stage.classes_:
        return "Unknown Stage"
    
    stage_encoded = le_stage.transform([stage_name])[0]
    subset = df[df["Crop Stage"] == stage_name]

    avg_start = subset["Days from Sowing (Start)"].mean()
    avg_duration = subset["Stage Duration (days)"].mean()

    features = pd.DataFrame([{
        "Days from Sowing (Start)": avg_start,
        "Stage Duration (days)": avg_duration,
        "Stage_encoded": stage_encoded
    }])

    pred = model.predict(features)[0]
    return le_disease.inverse_transform([pred])[0]


def predict_from_dates(sowing_date, current_date):
    sowing = datetime.strptime(sowing_date, "%Y-%m-%d")
    current = datetime.strptime(current_date, "%Y-%m-%d")

    days_after_sowing = (current - sowing).days
    stage = predict_stage(days_after_sowing)
    disease = predict_disease(stage)

    return days_after_sowing, stage, disease


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🌱 Cotton Crop Stage & Disease Prediction App")
st.write("Enter the sowing date and today's date to predict crop stage and possible diseases.")

sowing_date = st.date_input("Select Sowing Date")
current_date = st.date_input("Select Current Date")

if st.button("Predict"):
    sowing_str = sowing_date.strftime("%Y-%m-%d")
    current_str = current_date.strftime("%Y-%m-%d")

    days, stage, disease = predict_from_dates(sowing_str, current_str)

    st.success("Prediction Complete!")
    st.write(f"**Days After Sowing:** {days}")
    st.write(f"**Predicted Stage:** {stage}")
    st.write(f"**Predicted Disease:** {disease}")

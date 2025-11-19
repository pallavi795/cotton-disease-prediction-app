import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

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
    sow = datetime.strptime(sowing_date, "%Y-%m-%d")
    curr = datetime.strptime(current_date, "%Y-%m-%d")
    days = (curr - sow).days

    stage = predict_stage(days)
    disease = predict_disease(stage)

    return {
        "days_after_sowing": days,
        "predicted_stage": stage,
        "predicted_disease": disease
    }

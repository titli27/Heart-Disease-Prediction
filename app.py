import streamlit as st
import numpy as np
import joblib

import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")


age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

input_data = np.array([[float(age), float(sex), float(cp), float(trestbps),float(chol), float(fbs), float(restecg), float(thalach),float(exang), float(oldpeak), float(slope), float(ca), float(thal)]])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

if st.button("Predict"):
    
    prediction = model.predict(input_data)
    
    prediction = model.predict(input_scaled)
prob = model.predict_proba(input_scaled)

risk = prob[0][0] * 100

st.metric("Heart Disease Risk", f"{risk:.2f}%")
st.progress(int(risk))

import plotly.graph_objects as go

risk = prob[0][0] * 100

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk,
    title={'text': "Heart Disease Risk"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "red"},
        'steps': [
            {'range': [0,30], 'color': "green"},
            {'range': [30,60], 'color': "yellow"},
            {'range': [60,100], 'color': "red"}
        ]
    }
))

st.plotly_chart(fig)

import pandas as pd
import matplotlib.pyplot as plt

features = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak',
            'slope','ca','thal']

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.set_title("Feature Importance")
ax.invert_yaxis()

st.pyplot(fig)
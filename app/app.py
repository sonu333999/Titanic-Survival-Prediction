import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="Titanic AI Predictor", page_icon="🚢", layout="wide")

# ---------------- Background ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
    url("https://images.unsplash.com/photo-1529429611278-76b3b4b3bcb4");
    background-size: cover;
    color: white;
}
h1, h2, h3 {
    color: #00c6ff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PATH FIX (IMPORTANT) ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, 'models', 'titanic_model.pkl')
data_path = os.path.join(BASE_DIR, 'data', 'titanic.csv')

# ---------------- Load Model ----------------
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

model = joblib.load(model_path)

# ---------------- Title ----------------
st.markdown("<h1 style='text-align:center;'>🚢 Titanic Survival AI Predictor</h1>", unsafe_allow_html=True)
st.write("---")

# ---------------- Sidebar ----------------
st.sidebar.header("🧾 Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses", 0, 5, 0)
parch = st.sidebar.number_input("Parents/Children", 0, 5, 0)
fare = st.sidebar.number_input("Fare", 0.0, 500.0, 50.0)

sex_val = 0 if sex == "Male" else 1

# ---------------- Prediction ----------------
if st.sidebar.button("🔍 Predict"):

    data = np.array([[pclass, sex_val, age, sibsp, parch, fare]])

    prediction = model.predict(data)
    prob = model.predict_proba(data)[0][1] * 100

    st.write("## 🎯 Prediction Result")

    if prediction[0] == 1:
        st.success(f"🎉 Survived (Confidence: {prob:.2f}%)")
        st.balloons()
    else:
        st.error(f"💀 Did Not Survive (Confidence: {100-prob:.2f}%)")

# ---------------- EDA ----------------
st.write("---")
st.write("## 📊 Titanic Data Insights")

if not os.path.exists(data_path):
    st.error(f"Data file not found at: {data_path}")
    st.stop()

df = pd.read_csv(data_path)

col1, col2 = st.columns(2)

with col1:
    st.write("### Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("### Survival by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=df, ax=ax2)
    st.pyplot(fig2)

# ---------------- Footer ----------------
st.write("---")
st.caption("🚀 Built with Streamlit | Titanic ML Project")
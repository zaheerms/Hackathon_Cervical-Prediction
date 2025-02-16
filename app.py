import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load saved model and scaler
model = pickle.load(open("cervical_cancer_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# Title and Description
st.set_page_config(page_title="Cervical Cancer Prediction", layout="wide")
st.title("🩺 Cervical Cancer Prediction - AI Cure Innovators")
st.markdown("""
### 🚀 A cutting-edge AI-based cervical cancer prediction tool!
""")

# Team Members with Visuals
st.markdown("### 👥 Meet Our Team")
col1, col2, col3, col4 = st.columns(4)
col1.image("team_member1.jpg", width=100)
col1.markdown("**Mohamed Zaheerudeen**\nTeam Coordinator")
col2.image("team_member2.jpg", width=100)
col2.markdown("**Preethi Sherine**\nResearch Lead")
col3.image("team_member3.jpg", width=100)
col3.markdown("**Irsath Ahamed**\nDocumentation Specialist")
col4.image("team_member4.jpg", width=100)
col4.markdown("**Sam Daniel**\nProject Associate")

st.markdown("---")

# User Input in a Grid Layout
st.subheader("🔍 Patient Data Input")
columns = [
    "Age", "Number of Sexual Partners", "First Sexual Intercourse", "Num of Pregnancies",
    "Smokes", "Smokes (Years)", "Hormonal Contraceptives", "Hormonal Contraceptives (Years)",
    "IUD", "IUD (Years)", "STDs", "STDs (Number)", "STDs (HPV)", "STDs (Cervical Cancer)"
]

user_data = []
cols = st.columns(4)  # Create 4 columns for grid layout
for i, col in enumerate(columns):
    with cols[i % 4]:
        value = st.number_input(f"{col}", min_value=0.0, step=0.1)
        user_data.append(value)

# Convert input to array and Predict
if st.button("🔮 Predict Cancer Risk"):
    sample_data = np.array(user_data).reshape(1, -1)
    sample_scaled = scaler.transform(sample_data)
    prediction = model.predict(sample_scaled)[0]
    
    result = "🛑 **High Risk: Cancer Detected!**" if prediction == 1 else "✅ **Low Risk: No Cancer Detected**"
    st.markdown(f"### {result}")
    
    if prediction == 1:
        st.warning("Please consult a healthcare professional immediately.")
    else:
        st.success("No signs detected. Maintain regular health checkups!")

st.markdown("---")
st.markdown("📢 Developed by *AI Cure Innovators* for the Hackathon!")

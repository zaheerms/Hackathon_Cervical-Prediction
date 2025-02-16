import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load saved model and scaler
model = pickle.load(open("cervical_cancer_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# Title and Description
st.title("🩺 Cervical Cancer Prediction - AI Cure Innovators")
st.markdown("""
### Our Team:
- **Mohamed Zaheerudeen (Team Coordinator)**
- **Preethi Sherine (Research Lead)**
- **Irsath Ahamed (Documentation Specialist)**
- **Sam Daniel (Project Associate)**

🚀 *A cutting-edge AI-based cervical cancer prediction tool!* 
""")

# User Input Form
st.subheader("🔍 Patient Data Input")

# Define all 14 feature columns
columns = [
    "Age", "Number of Sexual Partners", "First Sexual Intercourse", "Num of Pregnancies", "Smokes", 
    "Smokes Years", "Hormonal Contraceptives", "Hormonal Contraceptives Years", "IUD", "IUD Years", 
    "STDs", "STDs Number", "STDs Condylomatosis", "STDs Time Since First Diagnosis"
]

user_data = []
for col in columns:
    value = st.number_input(f"Enter {col}", min_value=0.0, step=0.1)
    user_data.append(value)

# Convert input to array
if st.button("🔮 Predict Cancer Risk"):
    sample_data = np.array(user_data).reshape(1, -1)
    sample_scaled = scaler.transform(sample_data)
    prediction = model.predict(sample_scaled)[0]
    result = "Cancer Detected" if prediction == 1 else "No Cancer Detected"
    st.subheader(f"🧪 Prediction Result: {result}")

# Team Info & Footer
st.markdown("---")
st.markdown("📢 Developed by *AI Cure Innovators* for the Hackathon!")

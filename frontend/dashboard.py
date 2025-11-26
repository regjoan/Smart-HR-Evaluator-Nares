import streamlit as st
import pandas as pd
import os
import requests

# Page config
st.set_page_config(page_title="Smart HR Evaluator", layout="wide")
st.title("🌟 Smart HR Evaluator Dashboard")

# Load CSV
try:
    df = pd.read_csv('Employee 1000x.csv')
    st.sidebar.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("File not found. Please check the file path.")
    st.stop()

# --- MAIN TAB: Evaluate Employee ---
st.subheader("Evaluate Employee Performance")

with st.form("employee_form"):
    feature1 = st.number_input("Feature 1 (Task Completion Rate)", 0.0, 100.0, 0.0, 0.1)
    feature2 = st.number_input("Feature 2 (Collaboration Score)", 0.0, 100.0, 0.0, 0.1)
    feature3 = st.number_input("Feature 3 (Attendance Rate)", 0.0, 100.0, 0.0, 0.1)
    feature4 = st.number_input("Feature 4 (Quality of Work)", 0.0, 100.0, 0.0, 0.1)

    submitted = st.form_submit_button("Evaluate")

# Handle form submission
if submitted:
    employee_data = {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "feature4": feature4
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=employee_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Performance Score: {result['performance_score']}")
            st.info(f"Recommendation: {result['recommendation']}")
        else:
            st.error(f"Error: API returned {response.status_code}")

    except Exception as e:
        st.error(f"API connection error: {e}")

# Display data
st.subheader("Employee Data")
st.dataframe(df)

# Debug
st.write("Current working directory:", os.getcwd())

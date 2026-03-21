import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Drug Response Prediction", layout="centered")

# Sidebar
st.sidebar.title("About")
st.sidebar.write("Drug Response Classification using Machine Learning")
st.sidebar.write("Model: Support Vector Machine (SVM)")
st.sidebar.write("Accuracy: 0.78")
st.sidebar.write("AUC Score: 0.86")

# Title
st.title("Drug Response Classification App")
st.markdown("### Upload Patient Data or Enter Manually")
st.markdown("---")

# Load model
model = pickle.load(open("pipeline.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Sample CSV
sample_data = pd.DataFrame({
    "Blood Glucose Level (mg/dL)": [100],
    "Drug Dosage (mg)": [20],
    "Heart Rate (BPM)": [80],
    "Liver Toxicity Index (U/L)": [30],
    "Systolic Blood Pressure (mmHg)": [120]
})

st.download_button(
    label="Download Sample CSV",
    data=sample_data.to_csv(index=False),
    file_name="sample_drug_response.csv",
    mime="text/csv"
)

st.markdown("---")

# File upload
uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data")
    st.dataframe(data)

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    data["Prediction"] = predictions
    data["Probability"] = probabilities

    st.write("Prediction Results")
    st.dataframe(data)

    # Download predictions
    st.download_button(
        label="Download Predictions CSV",
        data=data.to_csv(index=False),
        file_name="drug_response_predictions.csv",
        mime="text/csv"
    )

st.markdown("---")

# Manual input
st.subheader("Manual Input")

glucose = st.number_input("Blood Glucose Level (mg/dL)")
dosage = st.number_input("Drug Dosage (mg)")
heart_rate = st.number_input("Heart Rate (BPM)")
liver = st.number_input("Liver Toxicity Index (U/L)")
bp = st.number_input("Systolic Blood Pressure (mmHg)")

if st.button("Predict Drug Response"):
    input_df = pd.DataFrame(
        [[glucose, dosage, heart_rate, liver, bp]],
        columns=features
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success("Positive Drug Response")
    else:
        st.error("No Drug Response")

    st.write("Prediction Probability:", probability)

st.markdown("---")
st.caption("Drug Response Prediction App | Machine Learning Project")
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import pandas as pd
import numpy as np
import plotly.express as px

from codebase.dashboard_graphs import MaternalHealthDashboard

warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore")
#change path to your files
maternal_model = pickle.load(open("E:/Major/maternal_care-main/notebook/model/finalized_maternal_model.sav", 'rb')) 
fetal_model = pickle.load(open("E:/Major/maternal_care-main/notebook/model/fetal_model.sav", 'rb'))

maternal_scaler = pickle.load(open("E:/Major/maternal_care-main/notebook/model/maternal_scaler.sav", 'rb'))
fetal_scaler = pickle.load(open("E:/Major/maternal_care-main/notebook/model/fetal_scaler.sav", 'rb'))

# Sidebar navigation
with st.sidebar:
    st.title("MedVerse AI")
    st.write("Welcome to MedVerse AI")
    st.write("Choose an option from the menu below to get started:")

    selected = option_menu('MedVerse AI',
                           ['Dashboard',
                            'Pregnancy Risk Prediction',
                            'Fetal Health Prediction',
                            'About us'],
                           icons=['chat-square-text', 'hospital', 'capsule-pill', 'clipboard-data'],
                           default_index=0)

# About us page
if selected == 'About us':
    st.title("Welcome to MedVerse AI")
    st.write("Our platform addresses maternal and fetal health, providing accurate predictions and proactive risk management.")
    col1, col2 = st.columns(2)
    with col1:
        st.header("1. Pregnancy Risk Prediction")
        st.write("Predict potential risks during pregnancy using parameters like age, blood sugar, and blood pressure.")
        st.image("graphics/pregnancy_risk_image.jpg", caption="Pregnancy Risk Prediction", use_container_width=True)
    with col2:
        st.header("2. Fetal Health Prediction")
        st.write("Analyze fetal well-being through CTG features such as accelerations, variability, and decelerations.")
        st.image("graphics/fetal_health_image.jpg", caption="Fetal Health Prediction", use_container_width=True)
    st.header("3. Dashboard")
    st.write("Monitor and manage health data with a user-friendly dashboard for predictive analyses.")

# Pregnancy Risk Prediction page
if selected == 'Pregnancy Risk Prediction':
    st.title('Pregnancy Risk Prediction')
    st.markdown("Predict pregnancy risk based on age, blood sugar, blood pressure, body temperature, and heart rate.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age of the Person')
        bodyTemp = st.text_input('Body Temperature in Celsius')
    with col2:
        diastolicBP = st.text_input('Diastolic BP in mmHg')
        heartRate = st.text_input('Heart rate in bpm')
    with col3:
        BS = st.text_input('Blood glucose in mmol/L')

    if st.button('Predict Pregnancy Risk'):
        try:
            input_data = np.array([[float(age), float(diastolicBP), float(BS), float(bodyTemp), float(heartRate)]])
            input_scaled = maternal_scaler.transform(input_data)
            predicted_risk = maternal_model.predict(input_scaled)[0]

            st.subheader("Risk Level:")
            if predicted_risk == 0:
                st.markdown('<p style="font-weight:bold;font-size:20px;color:green;">Low Risk</p>', unsafe_allow_html=True)
            elif predicted_risk == 1:
                st.markdown('<p style="font-weight:bold;font-size:20px;color:orange;">Medium Risk</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="font-weight:bold;font-size:20px;color:red;">High Risk</p>', unsafe_allow_html=True)
        except:
            st.error("Please enter valid numerical values for all fields.")

# Fetal Health Prediction page
if selected == 'Fetal Health Prediction':
    st.title('Fetal Health Prediction')
    st.markdown("Predict fetal health using cardiotocogram (CTG) parameters.")

    cols = st.columns(3)
    fetal_inputs = []
    fetal_labels = [
        'Baseline Value', 'Accelerations', 'Fetal Movement', 'Uterine Contractions',
        'Light Decelerations', 'Severe Decelerations', 'Prolongued Decelerations',
        'Abnormal Short Term Variability', 'Mean Value Of Short Term Variability',
        'Percentage Of Time With Abnormal Long Term Variability', 'Mean Value Long Term Variability',
        'Histogram Width', 'Histogram Min', 'Histogram Max', 'Histogram Number Of Peaks',
        'Histogram Number Of Zeroes', 'Histogram Mode', 'Histogram Mean', 'Histogram Median',
        'Histogram Variance', 'Histogram Tendency'
    ]

    # Create text inputs dynamically
    for i, label in enumerate(fetal_labels):
        col = cols[i % 3]
        fetal_inputs.append(col.text_input(label))

    if st.button('Predict Fetal Health'):
        try:
            fetal_array = np.array([[float(x) for x in fetal_inputs]])
            fetal_array_scaled = fetal_scaler.transform(fetal_array)
            predicted_risk = fetal_model.predict(fetal_array_scaled)[0]

            if predicted_risk == 0:
                st.markdown('<p style="font-weight:bold;font-size:20px;color:green;">Normal</p>', unsafe_allow_html=True)
            elif predicted_risk == 1:
                st.markdown('<p style="font-weight:bold;font-size:20px;color:orange;">Suspect</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="font-weight:bold;font-size:20px;color:red;">Pathological</p>', unsafe_allow_html=True)
        except:
            st.error("Please enter valid numerical values for all fields.")

# Dashboard page
if selected == "Dashboard":
    api_key = "579b464db66ec23bdd00000139b0d95a6ee4441c5f37eeae13f3a0b2"
    api_endpoint = f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"
    st.header("Dashboard")
    st.markdown("Interactive dashboard for maternal health metrics across regions.")

    dashboard = MaternalHealthDashboard(api_endpoint)
    dashboard.create_bubble_chart()
    with st.expander("Show More"):
        st.markdown(dashboard.get_bubble_chart_data())
    dashboard.create_pie_chart()
    with st.expander("Show More"):
        st.markdown(dashboard.get_pie_graph_data())

#Numpy Library
import numpy as np
#pickle library to load ML model
import pickle
#Dashboard Libraries
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

from sklearn.preprocessing import LabelEncoder

# Configuring Streamlit GUI

st.set_page_config(layout="wide")

#Menu options

selected = option_menu(None,
                           options = ["Home","Anomaly Detection"],
                           icons = ["house","trophy"],
                           default_index=0,
                           orientation="horizontal",
                           styles={"container": {"width": "100%"},
                                   "icon": {"color": "white", "font-size": "24px"},
                                   "nav-link": {"font-size": "24px", "text-align": "center", "margin": "-2px"},
                                   "nav-link-selected": {"background-color": "#6F36AD"}}) 
# # # MENU 1 - Home

if selected == "Home":
    col1,col2 = st.columns(2)
    with col1:
        st.write("Industrial anomaly detection is a critical component of modern industrial processes that involve the monitoring and analysis of data to identify abnormal behavior or deviations from expected patterns within industrial systems.")
        st.write("Anomaly detection problems have a great importance in industrial applications, because anomalies usually represent faults, failures or the emergence of such. To detect these automatically, Machine Learning algorithm is proposed for anomaly / fault detection and their classification")
        st.header("Tools and Technologies Used")
        st.write("Python,Streamlit,Numpy,Pandas,Scikit-learn,Matplotlib,Pickle,Seaborn")
        st.header("Machine Learning Algorithm")
        st.write("For Anomaly detection OneClassSVM and Logistic Regression is used and achieved good accuracy")
    with col2:
            st.image("ml.png")

#Menu-2 - Detection
if selected == "Anomaly Detection":
    with st.form("my_form"):
        temp = st.number_input("Enter Temperature")
        boiler = st.number_input("Enter Boiler_Name")
        submit_button = st.form_submit_button(label="DETECT ANOMALY")
    if submit_button:
        with open('anomaly_model.pkl', 'rb') as f:
            model = pickle.load(f)
        user_data=np.array([[boiler,temp]])
        y_pred = model.predict(user_data)
        status = y_pred[0]
        if status==0:
            st.write('## :green[The temperature of the boiler is classified as normal]')
            st.balloons()
        else:
            st.write('## :red[The temperature of the boiler is classified as abnormal] ')
            st.snow()
   

    
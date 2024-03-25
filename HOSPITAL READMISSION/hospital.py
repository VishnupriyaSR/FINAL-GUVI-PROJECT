#Numpy Library
import numpy as np
#pickle library to load ML model
import pickle
#Dashboard Libraries
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

# Configuring Streamlit GUI

st.set_page_config(layout="wide")

#Menu options

selected = option_menu(None,
                           options = ["Home","Readmission-Prediciton","Gradient-Boosting"],
                           icons = ["house","trophy","archive"],
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
        st.header("Objective")
        st.write("In this project, we aim to build a predictive model using machine learning algorithms that leverage patient demographic information, medical history, and clinical data to forecast the likelihood of hospital readmission within a specific time window")
        st.header("Problem Statement")
        st.write("Hospital readmission is a significant concern in healthcare systems worldwide, contributing to increased healthcare costs and patient morbidity. Predicting which patients are at higher risk of readmission can help healthcare providers allocate resources more efficiently and implement targeted interventions to reduce readmission rates")
     
        st.header("Approach")
        st.write("1.Data Collection and Preprocessing")
        st.write("2.EDA")
        st.write("3.Feature Engineering")
        st.write("4.Model Building and Evaluation")
        st.write("5.Model Deplyment")
                
        with col2:
             st.header("Introduction")
             st.write("Hospital readmission occurs when a patient is admitted to the hospital shortly after being discharged, often due to complications or exacerbation of their condition. Readmissions not only impose a financial burden on healthcare systems but also indicate gaps in patient care and management. Identifying patients at risk of readmission is crucial for healthcare providers to intervene early and prevent unnecessary hospitalizations.")
             st.header("Tools and Technologies Used")
             st.write("Python,Streamlit,Numpy,Pandas,Scikit-learn,Matplotlib,Pickle,Seaborn")

# # # MENU 2 - Readmission_Status-Prediction

 # Define the possible values for the dropdown menus
if selected=="Readmission-Prediciton":
    with st.form("my_form1"):
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.write("")
            id=st.text_input("Enter Patient-ID")
            age = st.text_input("Enter Age")
            gender= st.text_input("Enter Gender")
            disease= st.text_input("Enter Diagnosis")
                        
            
        with col3:               
            lab=st.text_input("Enter Number of Lab Procedures")
            in_patient=st.text_input("Enter Number of In-Patient Visist")
            medications=st.text_input("Enter Number of Medications")
            out_patient=st.text_input("Enter Number of Out-Patient Visits")
            emer_visit=st.text_input("Enter Number of Emergency Visits")
            num_diag=st.text_input("Enter Number of Diagnosis")
            
        submit_button = st.form_submit_button(label="HOSPITAL_READMISSION")
            
        if submit_button:
            
            with open('gradient_model.pkl', 'rb') as f:
                model = pickle.load(f)
            new_data_point = np.array([[age,lab,medications,out_patient,in_patient,emer_visit,num_diag]])
            #new_data_point = [[55, 1, 5, 1, 3, 2, 2]]
            y_pred = model.predict(new_data_point)
            
            if y_pred[0] == 1:
                st.write('Readmission')
            else:
                st.write('No Readmission')
                        
            

#Menu - 3 -GB
if selected == "Gradient-Boosting":
    col1,col2=st.columns(2)
    with col1:
        st.write("Gradient Boosting is a powerful ensemble learning technique used for classification and regression tasks. It builds multiple decision trees sequentially, where each tree corrects the errors of the previous one. The model combines weak learners (typically decision trees) to form a strong learner.")
        st.header("Overview")
        st.write("Initialization: The algorithm starts with an initial model, often a simple one like a single leaf.")
        st.write("Sequential Training of Weak Learners: In each iteration, a weak learner (decision tree) is added to the ensemble to correct the errors of the previous models. The weak learner is trained on the residuals (the differences between the predicted and actual values) of the previous model")
        st.write("Gradient Descent Optimization")
        st.write("Ensemble Combination: The predictions of all weak learners are combined to make the final prediction.")
        st.write("Advantages")
        st.write("1.It can capture complex nonlinear relationships in the data")
        st.write("2.It automatically handles feature interactions and variable importance")
        st.write("3.It's less prone to overfitting compared to other ensemble methods like Random Forest")
    
    with col2:
            st.image("gb.png")

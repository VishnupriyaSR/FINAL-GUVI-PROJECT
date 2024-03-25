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
                           options = ["Home","Loan-Default-Prediciton","Machine Learning"],
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
        st.write("Build a classification model to predict clients who are likely to default on their loan and give recommendations to the bank on the important features to consider while approving a loan.")
        st.header("Approach")
        st.write("1.Data Collection and Preprocessing")
        st.write("2.EDA")
        st.write("3.Feature Engineering")
        st.write("4.Model Building and Evaluation")
        st.write("5.Model Deplyment")
        st.header("Tools and Technologies Used")
        st.write("Python,Streamlit,Numpy,Pandas,Scikit-learn,Matplotlib,Pickle,Seaborn")
        st.header("Machine Learning Algorithm")
        st.write("For Loan-Default-Prediction,Random Forest is used and achieved 80% accuracy")
        with col2:
             st.image("loan.jpg")


#Menu - 3 -ML
if selected == "Machine Learning":
    col1,col2=st.columns(2)
    with col1:
        st.write("Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed.Machine Learning is a branch of artificial intelligence that develops algorithms by learning the hidden patterns of the datasets used it to make predictions on new similar type data, without being explicitly programmed for each task.")
        st.write("Machine learning is used in many different applications, from image and speech recognition to natural language processing, recommendation systems, fraud detection, portfolio optimization, automated task, and so on. ")
        st.write("Types of Machine Learning")
        st.write("1)Supervised Machine Learning   2)Unsupervised Machine Learning   3)Reinforcement Machine Learning")
        st.write("There are two main types of supervised learning:")
        st.write("1.Regression: Regression is a type of supervised learning where the algorithm learns to predict continuous values based on input features.The output labels in regression are continuous values.")
        st.write("2.Classification: Classification is a type of supervised learning where the algorithm learns to assign input data to a specific category or    class based on input features.The output labels in classification are discrete values.")
    with col2:
            st.image("ml.png")



# # # MENU 2 - Loan-Default-Prediction

 # Define the possible values for the dropdown menus
if selected=="Loan-Default-Prediciton":
    with st.form("my_form"):
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.write("")
            gender=["MALE","FEMALE"]
            employment=["EMPLOYED","UNEMPLOYED"]
            location=["URBAN","SUBURBAN","RURAL"]
            

            age = st.text_input("Enter Age")
            gender= st.selectbox("Gender", sorted(gender),key=2)
            gender_dict={"MALE":1,"FEMALE":0}
            income = st.text_input("Enter Income")
            employment = st.selectbox("Employment_Status", sorted(employment),key=3)
            employment_dict={"EMPLOYED":1,"UNEMPLOYED":0}
            location = st.selectbox("Location", sorted(location),key=4)
            location_dict={"RURAL":0,"SUBURBAN":1,"URBAN":2}
            
            
        with col3:               
            credit_score=st.text_input("Enter Credit_Score")
            debt=st.text_input("Enter Debt-to-Income-Ratio")
            balance=st.text_input("Enter Existing-Loan-Balance")
            amount=st.text_input("Enter Loan-Amount")
            rate=st.text_input("Enter Interest_Rate")
            months=st.text_input("Enter Loan_Duartion_Months")
            submit_button = st.form_submit_button(label="LOAN-DEFAULT-PREDICTION")
            
        if submit_button:
            with open('ranforest_model.pkl', 'rb') as f:
                model = pickle.load(f)
            #y_pred = model.predict(np.array([[56,1,39266,0,0,525,0.07,35197,9068,12,16]]))
            data=np.array([[age,gender_dict[gender],income,employment_dict[employment],location_dict[location],credit_score,debt,balance,amount,rate,months]])
            y_pred =  model.predict(data)
            if y_pred[0] == 1:
                st.write('Non-Default')
            else:
                st.write('Default')
            
            
                
               
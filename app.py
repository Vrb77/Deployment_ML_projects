import streamlit as st
import joblib
import pandas as pd

# set the tab title
st.set_page_config("Deployment Project")

# Set the page title
st.title("Loan Status Prediction Project")

# Set header
st.subheader("By Vaishnavi Badade")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("pre.joblib")
model = joblib.load("model_random.joblib")

# Create Input boxes that takes input from the user 
Age = st.number_input("Age",min_value=18,step=1)
Income = st.number_input("Income")
Home_Ownership = st.selectbox("Home Ownership",options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
Employee_length = st.number_input("Employee Length")
Loan_Intent = st.selectbox("Loan Intent",options=['EDUCATION','MEDICAL','PERSONAL','VENTURE','DEBTCONSOLIDATION','HOMEIMPROVEMENT'])
Loan_Grade = st.selectbox("Loan Grade",options=['B', 'C', 'A', 'D', 'E', 'F', 'G'])
Loan_Amount = st.number_input("Loan Amount")
Interest_Rate = st.number_input("Interset Rate")
Percent_Income = st.number_input("Percent Income")
Default_on_file = st.selectbox("Default On File",options=['N', 'Y'])
Credit_History = st.number_input("Credit History")

# Include a button. After providing all the inputs, user will click on the button. The button should provide the necessary predictions
submit = st.button("Predict Loan Status")

if submit:
    data = {
        'person_age':[Age],
        'person_income':[Income],
        'person_home_ownership':[Home_Ownership],
        'person_emp_length':[Employee_length],
        'loan_intent':[Loan_Intent],
        'loan_grade':[Loan_Grade],
        'loan_amnt':[Loan_Amount],
        'loan_int_rate':[Interest_Rate],
        'loan_percent_income':[Percent_Income],
        'cb_person_default_on_file':[Default_on_file],
        'cb_person_cred_hist_length':[Credit_History]
    }
    # Convert above dictionary into dataframe first
    xnew = pd.DataFrame(data)
    # Apply data cleaning and preprocessing on new data using pre pipeline
    xnew_pre = pre.transform(xnew)
    # predictions
    preds = model.predict(xnew_pre)
    if preds[0]==1:
        op = 'Loan Status is Approved'
        # st.subheader('Loan Status is Approved')
    else:
        op = 'Loan Status is Not Approved'
        # st.subheader('Loan Status is Not Approved')
    
    st.subheader(op)
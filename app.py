import mlflow.sklearn
import streamlit as st
from utilities import config
from collections import defaultdict

def get_model():
    # Load the registered model
    model_name = config.model_name
    model_version = config.version  # specify the version of the model you want to load
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    return model


def predict(model, values: list):
    import pandas as pd
    columns = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",    
        "remote_ratio",
        "company_size",
        "emp_residence_company_location"
    ]
    input_df = pd.DataFrame([values], columns=columns)
    return model.predict(input_df)


def create_streamlit_ui():
    st.sidebar.title("Model Prediction")
    st.sidebar.write(
        "This application allows you to predict salary based on various features."
    )

    st.title("Salary Prediction")
    st.markdown(
        """
        <style>
        .main {
            background-color: #333333;
            color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input fields for numerical and categorical inputs
    st.header("Input Features")
    st.subheader("Please provide the following details:")

    work_year = st.selectbox("Work Year", [2020, 2021, 2022], index=2)
    
    d1 = defaultdict(lambda : 'Others',{
        'Large':"L",
        'Small':"S",
        'Medium':"M"
    })
    
    d2 = defaultdict(lambda: "Others", {
        'Senior':"SE",
        'Mid-Level':"MI",
        'Entry-Level':"EN",
        'Executive':"EX"
    })
    
    d3 = defaultdict(lambda: "Others",{
        'Full-time':"FT",
        'Contract':"CT",
        'Freelance':"FL",
        'Part-Time':"PT"
    })
    
    experience_level = d2.get(st.selectbox("Experience Level", ["Senior", "Mid-Level", "Entry-Level", "Executive"], index=0))
    employment_type = d3.get(st.selectbox("Employment Type", ["Full-time", "Contract", "Freelance", "Part-Time"], index=0))
    job_title = st.selectbox(
        "Job Title",
        [
            "Data Engineer",
            "Data Scientist",
            "Data Analyst",
            "Machine Learning Engineer",
            "Analytics Engineer",
            "Others",
        ],
        index=0,
    )
    
    
    remote_ratio = st.select_slider("Remote %", options=[0, 50, 100], value=50)
    company_size = d1.get(st.radio("Company Size", ["Large", "Small", "Medium"], index=0))
    
    employee_residence = st.selectbox(
        "Employee Residence", ["US", "GB", "CA", "IN", "ES", "Others"], index=0
    )
    company_location = st.selectbox(
        "Company Location", ["US", "GB", "CA", "IN", "DE", "Others"], index=0
    )
    emp_residence_company_location = employee_residence + '_' + company_location

    if st.button("Predict"):
        model = get_model()
        input_data = [
            work_year,
            experience_level,
            employment_type,
            job_title,
            remote_ratio,
            company_size,
            emp_residence_company_location  
        ]
        prediction = predict(model, input_data)
        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")

create_streamlit_ui()

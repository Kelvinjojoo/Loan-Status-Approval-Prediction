# 🏦 Loan Status Approval Prediction

This is a simple web application built with **Streamlit** that predicts whether a loan application will be approved or not, based on applicant information such as income, credit history, person education, etc.

## 🔍 Overview

The purpose of this project is to demonstrate a machine learning model that can assist in automating loan application decisions using data driven insights. This is part of my Ujian Tengah Semester (UTS) project. It uses a classification model trained with `scikit-learn` and `xgboost`, and deployed with `Streamlit`.

## 🚀 Features

- Easy-to-use web interface built with **Streamlit**
- Predicts loan approval status: `Approved` or `Rejected`
- Based on real-world loan applicant data
- Fast and interactive model inference

## 📊 Input Features

The model considers the following inputs:

- `person_age`: Applicant's age
- `person_gender`: Applicant's gender
- `person_education`: Education level
- `person_income`: Annual income
- `person_emp_exp`: Years of employment experience
- `person_home_ownership`: Home ownership status
- `loan_amnt`: Loan amount requested
- `loan_intent`: Purpose of the loan
- `loan_int_rate`: Interest rate on the loan
- `loan_percent_income`: Loan amount as a percentage of income
- `cb_person_cred_hist_length`: Length of credit history (in years)
- `credit_score`: Credit score
- `previous_loan_defaults_on_file`: Whether previous loan defaults are on file (`yes` / `no`)

## 🎯 Target Variable

- `loan_status`: Loan approval status (`Approved` / `Rejected`)

## 🧠 Model

- The model is trained using a supervised learning algorithm (**XGBoost Classifier**).
- Data preprocessing includes handling missing values, encoding categorical variables, and scaling if needed. You can see how I'm doing preprocessing in `Evaluation_Model_Machine_Learning.ipynb`
- The trained model is saved using `pickle` for use in the app.

## 🌐 Try the Loan Approval Predictor 

You can test the model live via the Streamlit web app that I make here: 
https://loan-status-approval-prediction-kelvinjonathanyusach.streamlit.app/

## ⚙️ Deployment Notes

If you'd like to deploy this Streamlit app using the provided code, make sure to:

- Use **Python version 3.10** in your Streamlit deployment settings for compatibility.
- Ensure all required packages listed in `requirements.txt` are properly installed.



# 🏦 UTS - Loan Status Approval Prediction

This is a simple web application built with **Streamlit** that predicts whether a loan application will be approved or not, based on applicant information such as income, credit history, employment status, etc.

## 🔍 Overview

The purpose of this project is to demonstrate a machine learning model that can assist in automating loan application decisions using data-driven insights. This is part of a Ujian Tengah Semester (UTS) project.

## 🚀 Features

- Interactive user interface using **Streamlit**
- Predicts loan approval status (`Approved` / `Rejected`)
- Simple and fast deployment
- Built using **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**

## 📊 Input Features

The model considers the following inputs:

- Gender
- Married
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area

## 🧠 Model

- The ML model is trained on a loan dataset using classification algorithms (e.g., Logistic Regression or Random Forest).
- The model is serialized using `joblib` or `pickle` and loaded in the Streamlit app for real-time inference.

## 🖥️ How to Run the App Locally

1. **Clone the repo**

```bash
git clone https://github.com/your-username/UTS-Loan-Status-Approval-Prediction.git
cd UTS-Loan-Status-Approval-Prediction


import streamlit as st
import joblib
import pandas as pd

model= joblib.load("Pickle/xgb_model.pkl")

gender_encoder= joblib.load("Pickle/gender_encode.pkl")
previous_loan_encoder= joblib.load("Pickle/previous_loan_encode.pkl")
education_encoder= joblib.load("Pickle/person_education_encode.pkl")
home_ownership_encoder= joblib.load("Pickle/person_home_ownership_encode.pkl")
loan_intent_encoder= joblib.load("Pickle/loan_intent_encode.pkl")

age_scaler= joblib.load("Pickle/person_age_scaler.pkl")
income_scaler= joblib.load("Pickle/person_income_scaler.pkl")
emp_exp_scaler= joblib.load("Pickle/person_emp_exp_scaler.pkl")
loan_amnt_scaler= joblib.load("Pickle/loan_amnt_scaler.pkl")
int_rate_scaler= joblib.load("Pickle/loan_int_rate_scaler.pkl")
percent_income_scaler= joblib.load("Pickle/loan_percent_income_scaler.pkl")
cred_hist_scaler= joblib.load("Pickle/cb_person_cred_hist_length_scaler.pkl")
credit_score_scaler= joblib.load("Pickle/credit_score_scaler.pkl")


def convert_input_to_df(input_data):
  data= [input_data]
  df= pd.DataFrame(data, columns=[
    'person_age', 'person_gender', 'person_education', 'person_income','person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file'
  ])
  return df

def encode_features(df):
  df= df.replace(gender_encoder)
  df= df.replace(previous_loan_encoder)
  df= df.replace(education_encoder)

  home_ownership_encoded= home_ownership_encoder.transform(df[["person_home_ownership"]])
  home_ownership_df= pd.DataFrame(home_ownership_encoded.toarray(), columns= home_ownership_encoder.get_feature_names_out())
    
  loan_intent_encoded= loan_intent_encoder.transform(df[["loan_intent"]])
  loan_intent_df= pd.DataFrame(loan_intent_encoded.toarray(), columns=loan_intent_encoder.get_feature_names_out())
    
  df= pd.concat([df, home_ownership_df, loan_intent_df], axis= 1)
  df= df.drop(['person_home_ownership', 'loan_intent'], axis=1)
    
  return df

def scale_features(df):
  df["person_age"]= age_scaler.transform(df[["person_age"]])
  df["person_income"]= income_scaler.transform(df[["person_income"]])
  df["person_emp_exp"]= emp_exp_scaler.transform(df[["person_emp_exp"]])
  df["loan_amnt"]= loan_amnt_scaler.transform(df[["loan_amnt"]])
  df["loan_int_rate"]= int_rate_scaler.transform(df[["loan_int_rate"]])
  df["loan_percent_income"]= percent_income_scaler.transform(df[["loan_percent_income"]])
  df["cb_person_cred_hist_length"]= cred_hist_scaler.transform(df[["cb_person_cred_hist_length"]])
  df["credit_score"]= credit_score_scaler.transform(df[["credit_score"]])
    
  return df

def predict_loan_status(user_input):
  prediction= model.predict(user_input)
    
  return prediction[0]

def main():
  st.title('Loan Status Approval Prediction')
  st.info("This app predicts whether a loan application will be approved or not.")

  st.subheader("Applicant Information")
  with st.form("loan_form"):
    person_age= st.number_input("Age", min_value= 18, max_value= 140, value= 18)
    person_gender= st.selectbox("Gender", ["Male", "Female"])
    person_education= st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    person_income= st.number_input("Annual Income", min_value= 0, value= 0)
    person_emp_exp= st.number_input("Employment Experience (years)", min_value= 0, max_value= 120, value= 0)
    person_home_ownership= st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage", "Other"])
    loan_amnt= st.number_input("Loan Amount", min_value= 0, value= 0)
    loan_intent= st.selectbox("Loan Purpose", ["Debt Consolidation", "Home Improvement", "Venture", "Personal", "Education", "Medical"])
    loan_int_rate= st.number_input("Loan Interest Rate", min_value= 0.0, max_value= 30.0, value= 0.0)
    loan_percent_income= st.slider("Loan as Percentage of Income", 0.0, 1.0, 0.01)
    cb_person_cred_hist_length= st.number_input("Credit History Length (years)", min_value= 0, max_value= 50, value= 0)
    credit_score= st.number_input("Credit Score", min_value= 300, max_value= 900, value= 300)
    previous_loan_defaults= st.selectbox("Previous Loan Defaults", ["Yes", "No"])
    
    submitted= st.form_submit_button("Predict Loan Approval")
        
    if submitted:
      input_data= {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults
      }
            
      user_df= convert_input_to_df(input_data) 
            
      user_df= encode_features(user_df)
      user_df= scale_features(user_df)
            
      prediction= predict_loan_status(user_df)
            
      st.subheader("Prediction Results")
      if prediction == 1:
        st.success("✅ Loan Approved")
      else:
        st.error("❌ Loan Rejected")

if __name__ == "__main__":
  main()
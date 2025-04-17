import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import pickle

class Preprocessor:
  def __init__(self, filepath):
    self.filepath= filepath
    self.df= None
    self.x_train= None
    self.x_test= None
    self.y_train= None
    self.y_test= None
    self.category= []
    self.numerical= []
  
  def read_data(self):
    self.df= pd.read_csv(self.filepath)
  
  def split_data(self):
    x= self.df[self.df.columns.drop(['loan_status'])]
    y= self.df['loan_status']

    self.x_train, self.x_test, self.y_train, self.y_test= train_test_split(x, y, test_size= 0.2, random_state= 42)

  def null_values_handling(self):      
    self.x_train['person_income'].fillna(self.x_train['person_income'].median(), inplace=True)
    self.x_test['person_income'].fillna(self.x_train['person_income'].median(), inplace=True)

  def identify_column_types(self):
    for i in self.x_train.columns:
      if 'int' in str(self.x_train[i].dtype) or 'float' in str(self.x_train[i].dtype):
        self.numerical.append(i)
      else:
        self.category.append(i)
  
  def cleaning_categorical_data(self):
    self.x_train['person_gender']= self.x_train['person_gender'].str.lower().str.replace(" ", "").replace({"fe male": "female"})
    self.x_test['person_gender']= self.x_test['person_gender'].str.lower().str.replace(" ", "").replace({"fe male": "female"})
        
    self.x_train['loan_intent']= self.x_train['loan_intent'].replace({
      'DEBTCONSOLIDATION': 'DEBT CONSOLIDATION',
      'HOMEIMPROVEMENT': 'HOME IMPROVEMENT'
    })
    self.x_test['loan_intent']= self.x_test['loan_intent'].replace({
      'DEBTCONSOLIDATION': 'DEBT CONSOLIDATION',
      'HOMEIMPROVEMENT': 'HOME IMPROVEMENT'
    })
        
    for col in self.category:
      self.x_train[col]= self.x_train[col].astype(str).str.strip().str.title()
      self.x_test[col]= self.x_test[col].astype(str).str.strip().str.title()
  
  def encoding(self):
    gender_encoder= {"person_gender": {"Male": 1, "Female": 0}}
    self.x_train= self.x_train.replace(gender_encoder)
    self.x_test= self.x_test.replace(gender_encoder)
    pickle.dump(gender_encoder, open("gender_encode.pkl", "wb"))


    previous_loan_encoder= {"previous_loan_defaults_on_file": {"Yes": 1, "No": 0}}
    self.x_train= self.x_train.replace(previous_loan_encoder)
    self.x_test= self.x_test.replace(previous_loan_encoder)
    pickle.dump(previous_loan_encoder, open("previous_loan_encode.pkl", "wb"))


    person_education_encoder= {"person_education": {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}}
    self.x_train= self.x_train.replace(person_education_encoder)
    self.x_test= self.x_test.replace(person_education_encoder)
    pickle.dump(person_education_encoder, open("person_education_encode.pkl", "wb"))
    

    person_home_ownership_encoder= OneHotEncoder()
    person_home_ownership_train= pd.DataFrame(person_home_ownership_encoder.fit_transform(self.x_train[['person_home_ownership']]).toarray(), 
                             columns= person_home_ownership_encoder.get_feature_names_out())
    person_home_ownership_test= pd.DataFrame(person_home_ownership_encoder.transform(self.x_test[['person_home_ownership']]).toarray(), 
                            columns= person_home_ownership_encoder.get_feature_names_out())
    pickle.dump(person_home_ownership_encoder, open("person_home_ownership_encode.pkl", "wb"))


    loan_intent_encoder= OneHotEncoder()
    loan_intent_train= pd.DataFrame(loan_intent_encoder.fit_transform(self.x_train[['loan_intent']]).toarray(), 
                             columns= loan_intent_encoder.get_feature_names_out())
    loan_intent_test= pd.DataFrame(loan_intent_encoder.transform(self.x_test[['loan_intent']]).toarray(), 
                            columns= loan_intent_encoder.get_feature_names_out())
    pickle.dump(loan_intent_encoder, open("loan_intent_encode.pkl", "wb"))

    
    self.x_train= self.x_train.reset_index()
    self.x_test= self.x_test.reset_index()
        
    self.x_train= pd.concat([self.x_train, person_home_ownership_train, loan_intent_train], axis=1)
    self.x_test= pd.concat([self.x_test, person_home_ownership_test, loan_intent_test], axis=1)
        
    self.x_train= self.x_train.drop(['index', 'person_home_ownership', 'loan_intent'], axis=1)
    self.x_test= self.x_test.drop(['index', 'person_home_ownership', 'loan_intent'], axis=1)

  def scaling(self):
    for col in self.numerical:
      robust_scaler= RobustScaler()
      self.x_train[col]= robust_scaler.fit_transform(self.x_train[[col]])
      self.x_test[col]= robust_scaler.transform(self.x_test[[col]])
      pickle.dump(robust_scaler, open(f"{col}_scaler.pkl", "wb"))

class Modeling:
  def __init__(self, x_train, x_test, y_train, y_test):
    self.x_train= x_train
    self.x_test= x_test
    self.y_train= y_train
    self.y_test= y_test
    self.xgb_model= None
    self.y_pred_xgb= None

  def train_xgb(self):
    self.xgb_model= XGBClassifier(random_state= 42, n_estimators= 100, min_child_weight= 50, max_depth= 8)
    self.xgb_model.fit(self.x_train, self.y_train)
    self.y_pred_xgb= self.xgb_model.predict(self.x_test)
  
  def evaluation(self):    
    print("\nClassification Report XGBoost\n")
    print(classification_report(self.y_test, self.y_pred_xgb))

  def save_xgb_model(self):
    pickle.dump(self.xgb_model, open("xgb_model.pkl", "wb"))
   

preprocessor= Preprocessor('Dataset_A_loan.csv')
preprocessor.read_data()
preprocessor.split_data()
preprocessor.null_values_handling()
preprocessor.identify_column_types()
preprocessor.cleaning_categorical_data()
preprocessor.encoding()
preprocessor.scaling()

model= Modeling(preprocessor.x_train, preprocessor.x_test, preprocessor.y_train, preprocessor.y_test)

model.train_xgb()
model.evaluation()
model.save_xgb_model()
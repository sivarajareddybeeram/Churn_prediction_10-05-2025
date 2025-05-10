import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pickle
import tensorflow as tf

#load the training model
model=tf.keras.models.load_model('model.h5')

#load onehot encoder 
with open('onehot_encode_Geography.pkl','rb') as file:
    onehot_encode_Geography= pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)

## streamlit app
st.title("customer Churn Prediction")
#user inputs
geography= st.selectbox('Geography',onehot_encode_Geography.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
Age=st.slider('Age',18,100)
Balance=st.number_input("Balance")
Credit_score= st.number_input("Credit Score")
Estimated_salary= st.number_input("Estimated Salary")
Tenure=st.slider("Tenure",0,10)
Num_of_products=st.slider('Number of Products', 1,4)
Has_credit_card=st.selectbox('Has Credit Card',[0,1])
Is_Active_Member=st.selectbox('Is Active Member',[0,1])

# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode geography
geo_encoded = onehot_encode_Geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode_Geography.get_feature_names_out(['Geography']))

# Create input_data DataFrame
input_data = pd.DataFrame([[
    Credit_score, Age, Tenure, Balance, Num_of_products,
    Has_credit_card, Is_Active_Member, Estimated_salary, gender_encoded
]], columns=[
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender'
])

# Combine one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)



#scale the data 
print(scalar.feature_names_in_)
input_data = input_data[scalar.feature_names_in_]
input_data_scaled = scalar.transform(input_data)

#prediction churn
Prediction=model.predict(input_data_scaled)
Prediction_proba=Prediction[0][0]

st.write(f'churn Probability:{Prediction_proba:.2f}')

if Prediction_proba>0.5:
    st.write("customer going to Churn")
else:
    st.write("customer not going to churn")


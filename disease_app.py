#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Disease Type Prediction Application 
This app predicts the disease a person is having!


""")

st.sidebar.header('Input Features')

# Collect user input features 
def user_input_features():
    Age = st.sidebar.slider('Age', 15,74)# on peut mettre un valeur par défaut('Age', 15,74,40)
    Sex = st.sidebar.selectbox('Sex',('M','F'))
    Blood_P = st.sidebar.selectbox('Blood Pressur',('LOW','NORMAL','HIGH'))
    Cholesterol = st.sidebar.selectbox('Cholesterol',('NORMAL','HIGH'))
    So_Po = st.sidebar.slider('Sodium to potassium Ration in Blood', 6.27,38.2)
    data = {'Age': Age,
            'Sex': Sex,
            'Blood_P': Blood_P,
            'Cholesterol': Cholesterol,
            'So_Po': So_Po,}
    features = pd.DataFrame(data, index=[0])
    return features
        
input_df = user_input_features()


Disease_raw = pd.read_csv('C:/Users/ASUS/Desktop/streamlit/issue app/Disease.csv')
Disease = Disease_raw.drop(columns=['Disease'])
df = pd.concat([input_df,Disease],axis=0)

# Encoding of ordinal features
encode = ['Sex','Blood_P', 'Cholesterol']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Input features')
st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('Disease.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
Disease_type = np.array(['DiseaseA','DiseaseB','DiseaseC', 'DiseaseD', 'DiseaseE'])
st.write(Disease_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:





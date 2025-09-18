#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
drug = pd.read_csv('C:/Users/ASUS/Desktop/streamlit/issue app/Disease.csv')


df = drug.copy()
target = 'Disease'
encode = ['Sex', 'Blood_P', 'Cholesterol']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'DiseaseA':0, 'DiseaseB':1, 'DiseaseC':2, 'DiseaseD':3, 'DiseaseE':4}
def target_encode(val):
    return target_mapper[val]

df['Disease'] = df['Disease'].apply(target_encode)

# Separating X and y
X = df.drop('Disease', axis=1)
Y = df['Disease']

#  random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('Disease.pkl', 'wb'))


# In[ ]:

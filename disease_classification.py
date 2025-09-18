#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
pd.options.display.max_columns = None 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings 
warnings.filterwarnings('ignore')


# In[2]:


data= pd.read_csv('C:/Users/ASUS/Desktop/streamlit/issue app/Disease.csv')

# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


colonnes = data.columns.to_list()
colonnes


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.describe(include=['object'])


# In[9]:


features=['Age', 'Sex', 'Blood_P', 'Cholesterol', 'So_Po']


# In[10]:


X = data[features]
y = data['Disease']


# In[11]:


X.head()


# In[12]:


y.head()


# In[13]:


X = X.values
y = y.values
X[0:10,:]


# In[14]:


#encodage
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
X[:,1] = label.fit_transform(X[:,1])
X[:,3] = label.fit_transform(X[:,3])


# In[15]:


X[0:10,:]


# In[16]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('Blood_P',OneHotEncoder(),[2])],remainder = 'passthrough')


# In[17]:


X.shape


# In[18]:


X = ct.fit_transform(X)


# In[19]:


X.shape


# In[20]:


X[0:10,:]


# In[21]:


if X.shape[1] == 7:
    X = X[:,1:]


# In[22]:


X.shape


# In[23]:


X[0:10,:]


# In[24]:


data.columns


# In[25]:


features2=['Blood_P1', 'Blood_P2', 'Age', 'Sex', 'Cholesterol', 'So_Po']


# In[26]:


features2


# In[27]:


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state = 0)


# In[28]:


X_test[0:5,:]


# In[29]:


#standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[30]:


X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# In[31]:


X_train_sc[0,:]


# In[32]:


#gridsearch knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors' : [1,3,5,7,9,11,13,15]}
model = KNeighborsClassifier()
clf = GridSearchCV(model,parameters, scoring='accuracy', cv=5)
grille = clf.fit(X_train_sc,y_train)
print(grille.best_params_)
print(grille.best_score_)


# In[33]:


knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train_sc,y_train)


# In[34]:


y_pred_knn = knn.predict(X_test_sc)


# In[35]:


print('Confusion matrix knn \n', confusion_matrix(y_test,y_pred_knn))
print('Accuracy knn', accuracy_score(y_test,y_pred_knn))


# In[36]:


for i in range(10):
    print(y_test[i], y_pred_knn[i])


# In[37]:


print(classification_report(y_test,y_pred_knn))


# In[38]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[39]:


y_pred_dt = dt.predict(X_test)


# In[40]:


for i in range(10):
    print(y_test[i],y_pred_dt[i])


# In[41]:


print('Confusion matrix dt \n', confusion_matrix(y_test,y_pred_dt))
print('Accuracy dt', accuracy_score(y_test,y_pred_dt))


# In[42]:


import graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
model = DecisionTreeClassifier(max_depth = 5)
model.fit(X,y)


# In[43]:


tree.export_graphviz(model,feature_names = features2,\
                    out_file = 'dt_drug.dot',\
                    label = 'all',\
                    filled = True,\
                    rounded = True)


# In[44]:


from IPython.display import Image
Image('dt-drug.png')


# In[45]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)


# In[46]:


y_pred_rf = rf.predict(X_test)


# In[47]:


print('Confusion matrix rf \n', confusion_matrix(y_test,y_pred_rf))
print('Accuracy rf', accuracy_score(y_test,y_pred_rf))


# In[48]:


print(classification_report(y_test,y_pred_rf))


# In[49]:


#svm linear
from sklearn.svm import SVC
linear_SVM = SVC(kernel='linear')
linear_SVM.fit(X_train_sc,y_train)


# In[50]:


y_predictSVM_l = linear_SVM.predict(X_test_sc)
print(confusion_matrix(y_test,y_predictSVM_l))
print('Accuracy linear SVM {0:.3f}'.format(accuracy_score(y_test,y_predictSVM_l)))


# In[51]:


kernel_SVM = SVC(kernel='rbf')
kernel_SVM.fit(X_train_sc,y_train)
y_predictSVM_k = kernel_SVM.predict(X_test_sc)
print(confusion_matrix(y_test,y_predictSVM_k))
print('Accuracy rbf SVM {0:.3f}'.format(accuracy_score(y_test,y_predictSVM_k)))


# In[52]:


#logistic regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train_sc,y_train)


# In[53]:


y_predictLR = LR.predict(X_test_sc)
print(confusion_matrix(y_test,y_predictLR))
print('Accuracy Logistic Regression {0:.3f}'.format(accuracy_score(y_test,y_predictLR)))






#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


import plotly.offline as pyo
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
init_notebook_mode(connected=True)


# In[24]:


df = pd.read_excel('All_data_FIW_2013-2022.xlsx', sheet_name = 1, header=1)


# In[25]:


df.head()


# In[62]:


X = df.iloc[:, 7:-1]
X['Country/Territory'] = df['Country/Territory']
X = X.drop(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'CL', 'PR', 'Add Q', 'Add A'], axis=1)
X = X.to_numpy()
X


# In[65]:


y = df.iloc[:,4]
y


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X,y,  random_state = 1)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_pred = X_test.iloc[: , -1]


X_test = X_test.iloc[:, 0:-1]
X_train = X_train.iloc[:, 0:-1]
# X_train = X_train.drop(['Country/Territory'], axis=1)

X_test = X_test.to_numpy()
X_train = X_train.to_numpy()


# In[85]:


X_test


# In[86]:


y_test


# In[87]:


X_train = pd.DataFrame(X_train)
# X_train = pd.DataFrame(X_train).iloc[:, 0:-1]
X_train


# In[88]:


dummy = DummyClassifier()
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)

print("Baseline Classifier Accuracy:",metrics.accuracy_score(y_test, y_pred))

learner = SVC(kernel = 'linear',gamma='auto')
learner.fit(X_train, y_train)
y_pred = learner.predict(X_test)
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[89]:


y_pred


# In[90]:


learner = KNeighborsClassifier()

learner.fit(X_train, y_train)
y_pred = learner.predict(X_test)

print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[93]:


def plotPredictions(X_pred, y_pred):
    fig = px.choropleth(locations=X_pred,
     locationmode="country names",
    color=y_pred,
     title="Classification of Freedom State of Countries/Territories using SVM",
    labels={"color": "Election Type"}
     )
    fig.show()
    
plotPredictions(X_pred, y_pred)


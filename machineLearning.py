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


def setup():
    df = pd.read_excel('All_data_FIW_2013-2022.xlsx', sheet_name = 1, header=1)

    df.head()


    X = df.iloc[:, 3:-1]
    X['Country/Territory'] = df['Country/Territory']
    X = X.drop(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'CL', 'PR', 'Add Q', 'Add A', 'PR rating', 'CL rating'], axis=1)
    
    X_train = X[X['Country/Territory'] < 'R'] 
    X_train = X_train[X_train['Edition'] <= 2019]
    X_test = X[X['Country/Territory'] >= 'R'] 
    X_test = X_test[X_test['Edition'] > 2019]


    y_train = X_train.loc[:,"Status"]
    y_test = X_test.loc[:,"Status"]
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    X_pred = X_test.iloc[: , -1]
    X_train_pred = X_train.iloc[: , -1]

    X_test = X_test.iloc[:, 2:-1]
    X_train = X_train.iloc[:, 2:-1]

    X_test = X_test.to_numpy()
    X_train = X_train.to_numpy()
    return X_test, X_train, X_train_pred, X_pred, y_test, y_train


def predict_Baseline_Dummy():
    X_test, X_train, X_train_pred, X_pred, y_test, y_train = setup()

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)

    print("Baseline Classifier Accuracy:",metrics.accuracy_score(y_test, y_pred))

    


    
def predict_SVM():
    X_test, X_train, X_train_pred, X_pred, y_test, y_train = setup()

    learner = SVC(kernel = 'linear',gamma='auto')
    learner.fit(X_train, y_train)
    y_pred = learner.predict(X_test)
    print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return X_pred, y_pred


def predict_KNN():
    X_test, X_train, X_train_pred, X_pred, y_test, y_train = setup()
    
    learner = KNeighborsClassifier()

    learner.fit(X_train, y_train)
    y_pred = learner.predict(X_test)

    print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return X_pred, y_pred


def plotPredictions(X_pred, y_pred):
    fig = px.choropleth(locations=X_pred,
     locationmode="country names",
    color=y_pred,
     title="Wide-Form Input, relabelled",
    labels={"color": "Election Type"}
     )
    fig.show()
    
    

def plotTraining():
    X_test, X_train, X_train_pred, X_pred, y_test, y_train = setup()
#     X_train_pred
    fig = px.choropleth(locations=X_train_pred,
     locationmode="country names",
    color=y_train,
     title="Wide-Form Input, relabelled",
    labels={"color": "Election Type"}
     )
    fig.show()
    

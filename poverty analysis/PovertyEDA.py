#!/usr/bin/env python
# coding: utf-8

# # EDA for Multidimensional Poverty Index in developing countries and How it affects the freeness of a country
# 
# 
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[2]:


df = pd.read_excel("Multidimensional Poverty Index____.xls", header = 2, nrows = 102)
labels1 = pd.read_excel("Aggregate_Category_and_Subcategory_Scores_FIW_2003-2022.xlsx", sheet_name = 1)


# ### Data Cleaning for Multidimensional Poverty Index
# 
# 1. Rename useful column names which are unnamed to that present in the dataset
# 2. There exists an empty column 'b' filled with NAs so, delete that
# 3. Drop the first row as it has column names
# 4. Reset the index of the dataframe
# 5. Clean missing values

# In[3]:



df = df.rename(columns = {'Unnamed: 0':'Country', 'Unnamed: 18': 'Population vulnerable to multidimensional poverty'})

df = df.drop(df.filter(regex='Unnamed:').columns, axis=1)
df = df.drop('b', axis = 1)

df = df.drop(0)
df = df.reset_index(drop = True)

# replace missing values not represented by NaN with it
df = df.replace('..', np.NaN)

df


# ### Data Cleaning for Aggregate Category and Subcategory Scores
# 
# 1. Drop unnamed columns which don't have any data
# 2. Rename the column to match with that of the main dataframe
# 3. There is an empty coulmn 'x' which marks the last coulmn of the dataset, delete that
# 4. Get only the data from 2018 as the Multidimensional Poverty Index has data for that year
# 5. Reset the indices for readability

# In[4]:



labels = labels1.drop(labels1.filter(regex='Unnamed:').columns, axis=1)
labels = labels.rename(columns = {'Country/Territory': 'Country'})
labels = labels.drop(['x', 'Add Q', 'Add A'], axis = 1)
labels = labels[labels['Edition'] == 2018]
labels = labels.reset_index(drop = True)
labels.head(5)


# ### Transformation
# 
# Merge the two datasets to get the freedom rating for each of developing countries present in the multidimensional poverty index. This is done to get only the labels of each country and to explore the dataset of poverty in a way such that each country's statistics can be checked with only the freedom labels extracted from the other dataset.

# In[28]:


# def dataMerged():
    merged = pd.merge(df, labels[['Country', 'Status']], on =['Country'], how = 'left')
#     return merged.head(3)
# dataMerged()


# ## Visualizations for EDA
# 
# Based on the poverty index and the freedom rating of a country, we can do EDA to find meaningful relationships and trends among the two if there are any.

# ### Relationship of Freedom with Population Vulnerable to Poverty
# 
# This graph shows the distribution of population vulnerable to multidimensional poverty in each freedom category, i.e., it shows the distribution of the percent of population at risk of poverty in each category.
# 
# From the graph we see that the median range of population at risk for not free countries is the highest and 25% of the data lies between 15 and 20. The range of distribution for partially free countries is the highest followed by free countries. In not fre countries there is a higher percent of people susceptible to poverty as they are densely populated after the median whereas is in free countries it is much lower as it is equally distributed.

# In[6]:


y_axis_labels = ['Not Free', 'Partially Free', 'Free'] 

def vis1():
   ax = sns.catplot(
       y = 'Status', 
       x = 'Population vulnerable to multidimensional poverty', 
       data = merged, 
       kind = "box",
       orient= "h",
       palette = "flare",
   )

   ax.set(xlabel = "Percentage of Population vulnerable to multidimensional poverty", title = "Distribution of Population at Risk of Multidimensional Poverty Among Different Freedom Statuses")

   plt.yticks([0,1,2],y_axis_labels)

   ax.fig.set_figwidth(6)
   ax.fig.set_figheight(4)
   
vis1()


# ### Relationship of Freedom with Intensity of Deprivation
# 
# This graph explores the distribution of intensity of deprivation for poor people in each freedom category, i.e., it shows the distribution of the average deprivation score of health, education and standard of living in each freedom category for developing countries.
# 
# The graph shows that in countries that are not free there is a large range of intensity of deprivation and 50% of the data lies between 40 and 55. This means that the intensity of deprivation for poor people is the most in not free and partially free countries. Free countries have the smallest and least intensity of deprivation among poor people. This shows a clear relationship between the freedom of a country and how deprived the poor people are. There are a few outliers in free countries but the trend suggests that NF and PF countries are most deprived.

# In[7]:


def vis2():
    ax = sns.catplot(x = 'Status', y = 'Intensity of deprivation', data = merged, kind = "box", palette = "viridis")
    ax.set(title = "Distribution of Intensity of Deprivation Among Different Freedom Statuses", ylabel = "Distribution of Intensity of deprivation" )
    plt.xticks([0,1,2],y_axis_labels)
    ax.fig.set_figwidth(6)
    ax.fig.set_figheight(4)
vis2()


# ### Poverty in Countries Based On Their Freedom State
# 
# This graph shows how much percentage of the population among the various developing countries is in multidimensional poverty for the different freedom statuses.
# 
# The graph distinctly shows that the countries that are not free have the highest poverty rate and the countries that are free have the lowest with partially free moderate compared to the other two. This shows a relation between poverty and freeness of a country, especially for developing countries.

# In[8]:


def vis3():
    ax = sns.catplot(x = 'Status', y = 'Population in severe multidimensional poverty ', data = merged, kind = 'bar', palette = "ch:s=-.2,r=.6")
    ax.set(xlabel = "Percentage of Population in severe multidimensional poverty", title = "Distribution of Population in Severe Poverty Among Different Freedom Statuses")
    plt.xticks([0,1,2],y_axis_labels)
    ax.fig.set_figwidth(6)
    ax.fig.set_figheight(4)
vis3()


# ## Conclusion
# 
# From the above EDA a possible hypothesis is that the freeness of a country is dependent on the poverty of the country. Countries with a high poverty rate tend to be not free and thier poor people are intensively deprived of basic needs such as Education, Health and Standard of Living. 
# 

# In[9]:


def plotEDAvisualizations():
    vis1()
    vis2()
    vis3()
# plotEDAvisualizations()


# # Machine Learning

# In[10]:


mergedML = pd.merge(df, labels, on =['Country'], how = 'left')
mergedML = mergedML.drop(['C/T?', 'Edition', 'Total', 'Region', 'Year and survey'], axis = 1)
mergedML = mergedML.dropna()
mergedML.info()


# In[11]:


y = mergedML['Status']
# mergedX = mergedML.drop('Status',  axis = 1)

X_train, X_test, y_train, y_test = train_test_split(mergedML,y, test_size=0.2, random_state = 1)
# print(train.shape)
# print(test.shape)
# X_train = pd.DataFrame(X_train)
X_train = X_train.drop(['Status', 'Country'], 1)
X_test = X_test.drop(['Status', 'Country'], 1)
# y_train = np.array(y)

X_train
y_train.size
# X_train.dropna()


# In[17]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(kernel = 'linear',gamma='auto')))


# In[18]:


results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[19]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[20]:


print("Accuracy:", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[16]:


# X = np.array(train.drop(['Status', 'Country'], 1).astype(float))
# y = np.array(train['Status'])



# from sklearn.cluster import KMeans
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X_scaled)

# correct = 0
# for i in range(len(X)):
#     predict_me = np.array(X[i].astype(float))
#     predict_me = predict_me.reshape(-1, len(predict_me))
#     prediction = kmeans.predict(predict_me)
#     if prediction[0] == y[i]:
#         correct += 1

# # print(correct/len(X))


# In[ ]:





# In[ ]:





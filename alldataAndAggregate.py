import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

def readAggregateVis():
    df = pd.read_excel('Aggregate_Category_and_Subcategory_Scores_FIW_2003-2022.xlsx', sheet_name = 1, header=0)
    data = df.iloc[:, 0:19]
    data = data.drop(['Edition','Add Q', 'Add A'], axis = 1)
    avg = data.groupby(['Region']).mean()
    totalDf = pd.DataFrame(avg.loc[:,'Total'])
    return totalDf

def aggregateVis():
    df = pd.read_excel('Aggregate_Category_and_Subcategory_Scores_FIW_2003-2022.xlsx', sheet_name = 1, header=0)
    data = df.iloc[:, 0:19]
    data = data.drop(['Edition','Add Q', 'Add A'], axis = 1)
    avg = data.groupby(['Region']).mean()
    #totalTable = avg.loc[:,'Total']
    #print(totalTable)
    ax = avg.plot.bar(y='Total', rot=0, title="Average Total Agreggrate Score by Region")
    #avg.columns = ['Region','Total]
    #return avg

    
###########################################

def readFreedomStatuses():
    df = pd.read_excel('All_data_FIW_2013-2022.xlsx', sheet_name = 1, header=1)
    status_by_year = pd.DataFrame(df.groupby(['Edition','Status']).size())
    status_by_year.reset_index(inplace=True)
    status_by_year.columns = ['year', 'status', 'count']
    return status_by_year
    
def alldataVis():
    df = pd.read_excel('All_data_FIW_2013-2022.xlsx', sheet_name = 1, header=1)
    status_by_year = pd.DataFrame(df.groupby(['Edition','Status']).size())
    status_by_year.reset_index(inplace=True)
    status_by_year.columns = ['year', 'status', 'count']
    ax = sns.pointplot(x="year", y="count", hue="status", hue_order=['F','PF','NF'], data=status_by_year,
                      palette=['mediumseagreen','goldenrod','tomato'])
    ax.set(xlabel='Year', ylabel='Number of Countries/Territories',
           title='Number countries/territories considered Free, Partly Free and Not Free from 2013 to 2022')

    handlesObj, labels = ax.get_legend_handles_labels()

    plt.legend(labels=['Free','Partly Free','Not Free'], handles=handlesObj, loc=7,
               bbox_to_anchor=(1.3, .6), title='Freedom Status')
    
    
def readListDem():
    dem = pd.read_excel('List_of_Electoral_Democracies_FIW22.xlsx', sheet_name = 0, header=1)
    dem.columns = ['country', 'dem_elec']
    return dem

def mergingData():
    dem = pd.read_excel('List_of_Electoral_Democracies_FIW22.xlsx', sheet_name = 0, header=1)
    dem.columns = ['country', 'dem_elec']
    
    df = pd.read_excel('All_data_FIW_2013-2022.xlsx', sheet_name = 1, header=1)
    
    countries_22 = df[df['Edition'] == 2022]
    del countries_22['C/T']
    del countries_22['Region']
    del countries_22['Edition']
    del countries_22['Total']
    del countries_22['Add Q']
    del countries_22['Add A']
    countries_22.rename(columns = {'Country/Territory':'country'}, inplace = True)
    
    c_22 = countries_22.merge(dem[['country', 'dem_elec']], on = 'country', how = 'inner')
    
    return c_22

def ML():
    dem = pd.read_excel('List_of_Electoral_Democracies_FIW22.xlsx', sheet_name = 0, header=1)
    dem.columns = ['country', 'dem_elec']
    
    df = pd.read_excel('All_data_FIW_2013-2022.xlsx', sheet_name = 1, header=1)
    
    countries_22 = df[df['Edition'] == 2022]
    del countries_22['C/T']
    del countries_22['Region']
    del countries_22['Edition']
    del countries_22['Total']
    del countries_22['Add Q']
    del countries_22['Add A']
    countries_22.rename(columns = {'Country/Territory':'country'}, inplace = True)
    
    c_22 = countries_22.merge(dem[['country', 'dem_elec']], on = 'country', how = 'inner')
    
    X = c_22.iloc[:, 3:-1]
    y = c_22.iloc[:, -1]
    X.dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,  random_state = 1)
    
    X_train = pd.DataFrame(X_train)
    X_train = X_train.astype(float)
    X_train
    X_train.dropna()
    
    
    SVM = svm.LinearSVC()
    SVM.fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

 
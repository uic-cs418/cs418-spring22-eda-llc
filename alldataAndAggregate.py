import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def aggregateVis():
    df = pd.read_excel('Aggregate_Category_and_Subcategory_Scores_FIW_2003-2022.xlsx', sheet_name = 1, header=0)
    data = df.iloc[:, 0:19]
    data = data.drop(['Edition','Add Q', 'Add A'], axis = 1)
    avg = data.groupby(['Region']).mean()
    ax = avg.plot.bar(y='Total', rot=0, title="Average Total Agreggrate Score by Region")
    
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
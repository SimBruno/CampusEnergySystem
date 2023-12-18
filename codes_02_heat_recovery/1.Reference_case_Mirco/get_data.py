import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os



# path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','codes_01_energy_demand','Q_cluster_medium.csv'))
# data=pd.read_csv(path,index_col=None).rename(columns={'Unnamed: 0':'cluster'})
# print(data.sum(axis=1))
# path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','codes_01_energy_demand','clusters.csv'))
# cluster_data=pd.read_csv(path,index_col=None)
# print(cluster_data['hours'])


# # data.drop('Unnamed: 0',axis=1,inplace=True)
# # data['Q_tot']=data.loc[:, (data.columns != 'hours') & (data.columns != 'Irr') & (data.columns != 'Temp')].sum(axis=1)
# # print(data['hours'])
# # print(data['Q_tot'])

path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','codes_01_energy_demand','data_MOES.csv'))
data=pd.read_csv(path)
data=data.iloc[0:9]
path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','codes_01_energy_demand','clusters.csv'))
cluster_data=pd.read_csv(path,index_col=None)
print(data)
print(cluster_data)

h=[]
for j in cluster_data.values:
    
    x=data.apply(lambda x: x['Floor Area']*(x['K_th']*(21-j[0])-x['Spec Q_people']-0.8*x['Spec Elec']-x['K_sun']*j[1]/1000),axis=1).values
    x[x<0]=0
    h=h+[np.sum(x)*j[2]]

print(h)

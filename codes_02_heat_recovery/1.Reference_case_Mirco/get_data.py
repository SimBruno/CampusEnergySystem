import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os



path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','codes_01_energy_demand','Q_cluster_medium.csv'))
data=pd.read_csv(path,index_col=None).rename(columns={'Unnamed: 0':'cluster'})
print(data.sum(axis=1))
path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','codes_01_energy_demand','clusters.csv'))
cluster_data=pd.read_csv(path,index_col=None)
print(cluster_data['hours'])


#data.drop('Unnamed: 0',axis=1,inplace=True)
#data['Q_tot']=data.loc[:, (data.columns != 'hours') & (data.columns != 'Irr') & (data.columns != 'Temp')].sum(axis=1)
#print(data['hours'])
#print(data['Q_tot'])

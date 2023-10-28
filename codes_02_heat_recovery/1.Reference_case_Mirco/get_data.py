import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os


path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','codes_01_energy_demand','final_df_task1_medium.csv'))
data=pd.read_csv(path,index_col=None)
data.drop('Temp',axis=1,inplace=True)
data.drop('Irr',axis=1,inplace=True)
data.drop('Unnamed: 0',axis=1,inplace=True)
data[data<=0]=0
data['Q_tot']=data.loc[:, data.columns != 'hours'].sum(axis=1)
print(data['hours'])
print(data['Q_tot'])

import pandas as pd
import numpy as np
import os
import amplpy
from amplpy import AMPL, DataFrame
import matplotlib.pyplot as plt
import sys
import re
import glob


#def main(argc,argv):

# Parameters for sending data to ampl
query=['HP6_T','HP2_T','HP5_T','EPFL_IN_T','EPFL_OUT_T','Q_COND_LOAD','Q_EVAP_LOAD','W_HP_COMP1_POWER','W_HP_COMP2_POWER']
query_ampl=['T_cond','T_evap','T_hp_5','T_EPFL_in','T_EPFL_out','Q_cond','Q_evap','W_comp1','W_comp2']
dict_query={query[j]:query_ampl[j] for j in range(len(query))}

# Parameters for plotting
T_ext_span=np.linspace(0,30,100)
color=['r','b','g','k']

# Choice of fluid
fluid=['R290_LT','R290_MT','R1270_LT', 'R1270_MT']

# Initialization
results=pd.DataFrame(columns=['a','b','c','Cond_cost','Evap_cost', 'comp1_cost', 'comp2_cost','Total_cost'])
count=0
leg=[]

# Main loop
for j in fluid:

    # Getting values from reconciliation
    data_ampl=pd.DataFrame()
    num=re.search(r'\d+', j).group()
    temperature_type=j.split('_')[1]
    files=glob.glob(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','02_Reconciliation','Reconciled','reconc_r-'+num+"_"+temperature_type.lower()+'_'))+"*")
    for h in range(0,len(files)):
        data=pd.read_csv(files[h],sep='\t',skiprows=1, names=['tags','measured_value','measured_accuracy','reconciled_value','reconciled_accuracy','units','default']).drop(columns='default').set_index('tags')
        name=j+"_"+str(h+1)
        data_ampl[name]=data[data.index.isin(query)]['reconciled_value']

    # Cleaning table to send to ampl
    data_ampl=data_ampl.transpose()
    data_ampl=data_ampl.rename(columns=dict_query)
    data_ampl=data_ampl[query_ampl]
    data_ampl=pd.DataFrame(data=data_ampl.values,columns=query_ampl)
    data_ampl['Time']=range(1,len(files)+1)
    data_ampl.set_index('Time',inplace=True)

    # Getting values for ambient temperature
    ambiant_df=pd.read_csv(os.path.join(os.path.dirname( __file__ ),"Text.csv"))
    data_ampl['T_ext']=ambiant_df[j].dropna().values

    # Regression with ampl
    ampl=AMPL()
    ampl.read(os.path.join(os.path.dirname( __file__ ),"moes2021_P4_py.mod"))
    ampl.set_option("solver",'snopt')
    ampl.set_data(data_ampl,"Time")
    ampl.solve()

    # Retrieving results
    a=ampl.get_variable("a").get_values().to_list()[0]
    b=ampl.get_variable("b").get_values().to_list()[0]
    c=ampl.get_variable("c").get_values().to_list()[0]
    Cond_cost=ampl.get_variable("Cond_cost").get_values().to_list()[0]
    Evap_cost=ampl.get_variable("Evap_cost").get_values().to_list()[0]
    comp1_cost=ampl.get_variable("comp1_cost").get_values().to_list()[0]
    comp2_cost=ampl.get_variable("comp2_cost").get_values().to_list()[0]
    Total_cost=Cond_cost+Evap_cost+comp1_cost+comp2_cost
    T_ext=[val[1] for val in ampl.get_parameter("T_ext").get_values().to_list()]
    carnot=[val[1] for val in ampl.get_variable("c_factor1").get_values().to_list()]

    # Saving values
    results.loc[j]=pd.Series({'a':a,'b':b,'c':c,'Cond_cost':Cond_cost,'Evap_cost': Evap_cost,'comp1_cost':comp1_cost,'comp2_cost':comp2_cost,'Total_cost':Total_cost})
    
    # Plotting results
    carnot_span=a*np.power(T_ext_span,2)-b*T_ext_span+c
    plt.plot(T_ext_span,carnot_span,lw=2,c=color[count])
    plt.scatter(T_ext,carnot,c=color[count])
    leg+=["Fitted "+j]+['Data '+j]
    count+=1

# Print results
print(results)

# Plot
plt.title("Carnot efficiency function of ambient temperature")
plt.xlabel("Ambient temperature [°C]")
plt.ylabel("Carnot efficiency [-]")  
plt.grid()
plt.xlim([min(T_ext_span),max(T_ext_span)])
plt.legend(leg)
plt.show()

# if __name__=="__main__":
#     try:
#         main(len(sys.argv),sys.argv)
#     except Exception as e:
#         print(e)
#         raise
import pandas as pd
import numpy as np
import os
import amplpy
from amplpy import AMPL, DataFrame
import matplotlib.pyplot as plt
import sys


#def main(argc,argv):
#####Set parameters######
number_of_files=10
temperature_type='lt' #'lt' for low temperature, 'mt' for medium temperature
#########################

query=['HP6_T','HP2_T','HP5_T','EPFL_IN_T','EPFL_OUT_T','Q_COND_LOAD','Q_EVAP_LOAD','W_HP_COMP1_POWER','W_HP_COMP2_POWER']
query_ampl=['T_cond','T_evap','T_hp_5','T_EPFL_in','T_EPFL_out','Q_cond','Q_evap','W_comp1','W_comp2']
dict_query={query[j]:query_ampl[j] for j in range(len(query))}
data_ampl=pd.DataFrame()#data=query_ampl,columns=['tags']).set_index('tags')
for j in range(1,number_of_files+1):
    name=temperature_type+'_'+str(j)
    filename='reconc_r-290_'+name+'.txt'
    path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','02_Reconciliation','Reconciled',filename))
    data=pd.read_csv(path,sep='\t',skiprows=1, names=['tags','measured_value','measured_accuracy','reconciled_value','reconciled_accuracy','units','default']).drop(columns='default').set_index('tags')
    data_ampl[name]=data[data.index.isin(query)]['reconciled_value']

data_ampl=data_ampl.transpose()
data_ampl=data_ampl.rename(columns=dict_query)
data_ampl=data_ampl[query_ampl]
data_ampl=pd.DataFrame(data=data_ampl.values,columns=query_ampl)
data_ampl['Time']=range(1,number_of_files+1)
data_ampl.set_index('Time',inplace=True)

ambiant_df=pd.read_csv(os.path.join(os.path.dirname( __file__ ),"Text.csv"))

data_ampl['T_ext']=ambiant_df['R290_LT'].dropna().values
print(data_ampl)

ampl=AMPL()
ampl.read(os.path.join(os.path.dirname( __file__ ),"moes2021_P4_py.mod"))
ampl.set_option("solver",'snopt')
ampl.set_data(data_ampl,"Time")
ampl.solve()
a=ampl.get_variable("a").get_values().to_list()[0]
b=ampl.get_variable("b").get_values().to_list()[0]
c=ampl.get_variable("c").get_values().to_list()[0]

T_ext=[val[1] for val in ampl.get_parameter("T_ext").get_values().to_list()]
carnot=[val[1] for val in ampl.get_variable("c_factor1").get_values().to_list()]

print(carnot)
Cond_cost=ampl.get_variable("Cond_cost").get_values().to_list()[0]
Evap_cost=ampl.get_variable("Evap_cost").get_values().to_list()[0]
comp1_cost=ampl.get_variable("comp1_cost").get_values().to_list()[0]
comp2_cost=ampl.get_variable("comp2_cost").get_values().to_list()[0]
Total_cost=Cond_cost+Evap_cost+comp1_cost+comp2_cost


print("Regression factors: a={:e}, b={:e}, c={:e}".format(a,b,c))
print("Condenser cost: {:e} [CHF]".format(Cond_cost))
print("Evaporator cost: {:e} [CHF]".format(Evap_cost))
print("First compressor cost: {:e} [CHF]".format(comp1_cost))
print("Second compressor cost: {:e} [CHF]".format(comp2_cost))
print("Total cost: {:e}".format(Total_cost))
T_ext_span=np.linspace(0,30,100)
carnot_span=a*np.power(T_ext_span,2)+b*T_ext_span+c
plt.plot(T_ext_span,carnot_span,lw=2)
plt.title("Carnot efficiency function of ambient temperature")
plt.xlabel("Ambient temperature [°C]")
plt.ylabel("Carnot efficiency [-]")
plt.scatter(T_ext,carnot,c='r')
plt.legend(["Regression","Observed data"])
plt.grid()
plt.show()



# if __name__=="__main__":
#     try:
#         main(len(sys.argv),sys.argv)
#     except Exception as e:
#         print(e)
#         raise

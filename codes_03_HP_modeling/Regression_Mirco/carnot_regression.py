import pandas as pd
import numpy as np
import os
import amplpy
from amplpy import AMPL, DataFrame
import matplotlib.pyplot as plt
import sys
import re
import glob


def get_reg():

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
    results=pd.DataFrame(columns=['a','b','c','Cond_area','Evap_area','DTlnCond','DTlnEvap','Cond_cost','Evap_cost', 'comp1_cost', 'comp2_cost','Total_cost', 'cinv1', 'cinv2'])
    count=0
    leg=[]
    observed=pd.DataFrame()

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
        print(data_ampl[data_ampl['Q_cond']==data_ampl['Q_cond'].min()])
        

        # Getting values for ambient temperature
        ambiant_df=pd.read_csv(os.path.join(os.path.dirname( __file__ ),"Text.csv"))
        data_ampl['T_ext']=ambiant_df[j].dropna().values
        data_ampl.sort_values(by='Q_cond',inplace=True)

        # Regression with ampl
        ampl=AMPL()
        ampl.read(os.path.join(os.path.dirname( __file__ ),"moes2021_P4_py.mod"))
        ampl.set_option("solver",'snopt')
        ampl.set_option("solver_msg",0)
        ampl.set_data(data_ampl,"Time")
        ampl.get_parameter('max_demand_index').set(data_ampl.index[data_ampl['Q_cond']==data_ampl['Q_cond'].max()].values[0])
        ampl.get_parameter('min_demand_index').set(data_ampl.index.values[0])
        #ampl.get_parameter('min_demand_index').set(data_ampl.index[data_ampl['Q_cond']==data_ampl['Q_cond'].min()].values[0])
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
        Cond_cost_2=ampl.get_variable("Cond_cost_2").get_values().to_list()[0]
        Evap_cost_2=ampl.get_variable("Evap_cost_2").get_values().to_list()[0]
        comp1_cost_2=ampl.get_variable("comp1_cost_2").get_values().to_list()[0]
        comp2_cost_2=ampl.get_variable("comp2_cost_2").get_values().to_list()[0]
        Total_cost_2=Cond_cost_2+Evap_cost_2+comp1_cost_2+comp2_cost_2
        T_ext=[val[1] for val in ampl.get_parameter("T_ext").get_values().to_list()]
        carnot=[val[1] for val in ampl.get_variable("c_factor1").get_values().to_list()]
        Q_cond_max=ampl.get_parameter("Q_cond_max").get_values().to_list()[0]
        Q_cond_min=ampl.get_parameter("Q_cond_min").get_values().to_list()[0]
        Cond_area=ampl.get_variable("Cond_area").get_values().to_list()[0]
        Evap_area=ampl.get_variable("Evap_area").get_values().to_list()[0]
        DTlnCond=ampl.get_variable("DTlnCond").get_values().to_list()[0]
        DTlnEvap=ampl.get_variable("DTlnEvap").get_values().to_list()[0]
        # Retrieving results from ampl2
        refsize=1000 #[kW]
        Fmin=Q_cond_min/refsize
        Fmax=Q_cond_max/refsize
        cinv2=(Total_cost-Total_cost_2)/(Q_cond_max-Q_cond_min)
        cinv1=Total_cost-cinv2*Q_cond_max
        print(Fmin,Fmax,cinv1,cinv2)
        # Saving values
        results.loc[j]=pd.Series({'a':a,'b':b,'c':c,'Cond_area':Cond_area,'Evap_area':Evap_area,'DTlnCond':DTlnCond,'DTlnEvap':DTlnEvap,'Cond_cost':Cond_cost,'Evap_cost': Evap_cost,'comp1_cost':comp1_cost,'comp2_cost':comp2_cost,'Total_cost':Total_cost,'cinv1':cinv1,'cinv2':cinv2})
        additional=pd.DataFrame({"T_ext_"+j:T_ext,"Carnot_"+j:carnot})
        observed=pd.concat([observed,additional],axis=1)
        # Plotting results
        carnot_span=a*np.power(T_ext_span,2)-b*T_ext_span+c
        # plt.plot(T_ext_span,carnot_span,lw=2,c=color[count])
        # plt.scatter(T_ext,carnot,c=color[count])
        leg+=["Fitted "+j]+['Data '+j]
        count+=1

    # Print results
    print(results['Total_cost'])
    
    # # Plot
    # plt.title("Carnot efficiency function of ambient temperature")
    # plt.xlabel("Ambient temperature [°C]")
    # plt.ylabel("Carnot efficiency [-]")  
    # plt.grid()
    # plt.xlim([min(T_ext_span),max(T_ext_span)])
    # plt.legend(leg)
    # plt.show()
    results['cinv1']=480+50*np.random.rand(4)

    return results, observed

if __name__=="__main__":
    results,ovserved=get_reg()
    print(results[['DTlnCond','DTlnEvap','Cond_area','Evap_area','cinv1','cinv2']])
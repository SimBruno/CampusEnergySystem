import pickle
import os
import amplpy
from amplpy import AMPL, Environment
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


def save_ampl_results(ampl, pkl_name="results"): 

  results = {}

  for o in ampl.getObjectives():
    values = o[1].getValues().toList()[0]
    results[o[1].name()] = values

  for v in ampl.getVariables():
    values = ampl.getData(v[1].name()).toPandas()
    results[v[1].name()] = values

  for p in ampl.getParameters():
    values = ampl.getData(p[1].name()).toPandas()
    results[p[1].name()] = values

  # save data
  #result_file_path = "./codes_04_energy_system_integration/results/"+ pkl_name + ".pkl"
  result_file_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ),'results',pkl_name+".pkl"))
  #result_file_path = "./codes_04_energy_system_integration/results/scenario1.pkl"
  f = open(result_file_path, 'wb')
  pickle.dump(results, f)
  f.close()
  return 


class criteria(Enum):
    OPEX = 'OPEX'
    CAPEX = 'CAPEX'
    Emissions = 'Emissions'
    TOTEX = 'TOTEX'
    parametric = 'parametric'


def optimize(criteria,result_file="optimize_dump",remove_tech=[],TAX=120e-6,Max_Emissions=1e30, Max_Totalcost=1e30, Max_Invcost=1e30,Max_Opcost=1e30,NatGasGrid=0.0303,ElecGridSell=-0.06,ElecGridBuy=0.0916,HydrogenGrid=0.3731, c_gas=228, c_elec=75.3):
  """Optimize the EPFL energy system.
  
  First specifiy criteria by writing for example criteria.OPEX, criteria.parametric, or criteria.Emissions.
  By default, TOTEX is optimized. If criteria.parametric is selected, a tax is applied to CO2 emissions, and the TOTEX is optimized. You can specify the CO2tax by setting TAX. by default, TAX=120e-6 [CHF/tCO2].
  
  A constraint on the maximum value of the TOTEX,CAPEX,OPEX and Emissions can be specified via
  Max_Totalcost, Max_Invcost, Max_Opcost, Max_Emissions. By default, they are unconstrained (more precisely constrained to 1e20 [gCO2/yr], which is so large that it won't affect the optimization).
  
  You can chose to remove a technology from the model by including it in the string list remove_tech. For exemple: remove_tech=['SOFC','STC']. You can remove any technology from this list:'Boiler', 'PV', 'SOFC', 'STC', 'R1270_LT', 'R1270_MT', 'R290_LT', 'R290_MT'. By default, all the technologies are included.

  The price of resources can be specified using NatGasGrid, ElecGridBuy, ElecGridSell, HydrogenGrid. By default, resources have a price of
  NatGasGrid=0.0303 [CHF/kWh], ElecGridBuy=0.0916 [CHF/kWh], ElecGridSell=-0.06 [CHF/kWh], HydrogenGrid=0.3731 [CHF/kWh].

  Lastly, the emission factor for electricity and gas can be changed using c_elec and c_gas. by default, they are at c_elec=75.3 [CHF/tCO2], c_gas=228 [gCO2/kWh]

  The output is also saved to pickle format in the folder results. The name of the file can be specified using result_file. By default, result_file="optimize_dump".
  """
  
  # Initialize ampl
  ampl = AMPL()
  # Go in right directory
  ampl.cd("./codes_04_energy_system_integration/ampl_files")

  # Read main .mod file 
  ampl.read("moes.mod")

  # Select right optimization criteria
  if criteria==criteria.Emissions:
    ampl.read("objective_Emissions.mod")
  elif criteria==criteria.OPEX:
    ampl.read("objective_OPEX.mod")
  elif criteria==criteria.CAPEX:
    ampl.read("objective_CAPEX.mod")
  elif criteria==criteria.parametric:
    ampl.read("objective_parametric.mod")
  else:
    ampl.read("objective_TOTEX.mod")
  
  # Construct the string lists of all sets
  Technologies = ['Boiler', 'PV', 'SOFC', 'STC', 'R1270_LT', 'R1270_MT', 'R290_LT', 'R290_MT']
  Grids=['ElecGridBuy','ElecGridSell', 'NatGasGrid','HydrogenGrid']
  Layers=['Natgas', 'Electricity', 'Hydrogen']
  UtilityType=['Heating', 'ElectricitySup', 'ElectricityCons']
  UtilitiesOfType_Heating=['Boiler','SOFC', 'STC', 'R1270_LT', 'R1270_MT', 'R290_LT', 'R290_MT']
  UtilitiesOfType_ElectricitySup=['SOFC', 'PV', 'ElecGridBuy']
  UtilitiesOfType_ElectricityCons=['R1270_LT', 'R1270_MT', 'R290_LT', 'R290_MT', 'ElecGridSell']
  UtilitiesOfLayer_Electricity=['SOFC', 'PV', 'R1270_LT', 'R1270_MT', 'R290_LT', 'R290_MT', 'ElecGridBuy', 'ElecGridSell'] 
  UtilitiesOfLayer_Natgas=['Boiler', 'NatGasGrid']
  UtilitiesOfLayer_Hydrogen=['SOFC','HydrogenGrid']
  HP2 = ['R1270_LT', 'R1270_MT', 'R290_LT', 'R290_MT']

  # Set the generic sets
  ampl.set['Grids'] = Grids
  ampl.set['Layers'] = Layers
  ampl.set['UtilityType'] = UtilityType

  # Remove the technology to remove from all sets
  for x in remove_tech:
    if x in Technologies:
      Technologies.remove(x)
    if x in UtilitiesOfType_Heating:
      UtilitiesOfType_Heating.remove(x)
    if x in UtilitiesOfType_ElectricityCons:
      UtilitiesOfType_ElectricityCons.remove(x)
    if x in UtilitiesOfType_ElectricitySup:
      UtilitiesOfType_ElectricitySup.remove(x)
    if x in UtilitiesOfLayer_Electricity:
      UtilitiesOfLayer_Electricity.remove(x)
    if x in UtilitiesOfLayer_Hydrogen:
      UtilitiesOfLayer_Hydrogen.remove(x)
    if x in UtilitiesOfLayer_Natgas:
      UtilitiesOfLayer_Natgas.remove(x)
    if x in HP2:
      HP2.remove(x)
    
  # Set all sets
  ampl.set['Technologies'] =Technologies
  ampl.set['HP2'] =HP2
  ampl.set['UtilitiesOfType']['Heating']=UtilitiesOfType_Heating
  ampl.set['UtilitiesOfType']['ElectricitySup']=UtilitiesOfType_ElectricitySup
  ampl.set['UtilitiesOfType']['ElectricityCons']=UtilitiesOfType_ElectricityCons
  ampl.set['UtilitiesOfLayer']['Electricity']=UtilitiesOfLayer_Electricity
  ampl.set['UtilitiesOfLayer']['Natgas']=UtilitiesOfLayer_Natgas
  ampl.set['UtilitiesOfLayer']['Hydrogen']=UtilitiesOfLayer_Hydrogen


  # Read main .dat file (just for the flows, everything else was set through python)
  ampl.readData("moes.dat")

  # Read all other Technology files, if it is not asked to remove them
  
  if 'Boiler' not in remove_tech:
    ampl.read("moesboiler.dat")
  if 'R290_LT' not in remove_tech:
    ampl.read("moesHP_R290_LT.mod")
    ampl.read("moesHP_R290_LT.dat")
  if 'R290_MT' not in remove_tech:
    ampl.read("moesHP_R290_MT.mod")
    ampl.read("moesHP_R290_MT.dat")
  if 'R1270_LT' not in remove_tech:
    ampl.read("moesHP_R1270_LT.mod")
    ampl.read("moesHP_R1270_LT.dat")
  if 'R1270_MT' not in remove_tech:
    ampl.read("moesHP_R1270_MT.mod")
    ampl.read("moesHP_R1270_MT.dat")
  if 'SOFC' not in remove_tech:
    ampl.read("moesSOFC.dat")
  if 'STC' not in remove_tech:
    ampl.read("moesSTC.mod")
    ampl.read("moesSTC.dat")
  if 'PV' not in remove_tech:
    ampl.read("moesPV.mod")
    ampl.read("moesPV.dat")

  # Set ampl options
  ampl.setOption('solver', 'gurobi')
  ampl.setOption('presolve_eps', 5e-05)
  ampl.setOption('omit_zero_rows', 1)
  ampl.setOption('omit_zero_cols', 1)
  
  # Import cluster data
  data_cluster=pd.read_csv("./codes_01_energy_demand/clusters_data.csv").drop(columns='Q_th').reset_index().rename(columns={'Temp':'Text', 'Irr':'irradiation', 'Hours':'top','index':'Time'}).set_index('Time')
  data_cluster['irradiation']=data_cluster['irradiation']/1000
  
  # Set cluster data
  ampl.set_data(data_cluster, "Time")

  # Read data from task 1
  data=pd.read_csv("./codes_01_energy_demand/thermal_properties.csv")
  data.index = ["Building" + str(i) for i in range(1,len(data)+1)] # the index of the dataframe has to match the values of the set "Buildings" in ampl

  # Set Buildings parameter
  for col in data.columns:
    ampl.getParameter(col).setValues(data[col])
  
  # Set CO2TAX and emission factors
  ampl.get_parameter("CO2tax").set(TAX)
  ampl.get_parameter("c_gas").set(c_gas)
  ampl.get_parameter("c_elec").set(c_elec)

  # Set Maximum values for criterias
  ampl.get_parameter("Max_Emissions").set(Max_Emissions)
  ampl.get_parameter("Max_Totalcost").set(Max_Totalcost)
  ampl.get_parameter("Max_Invcost").set(Max_Invcost)
  ampl.get_parameter("Max_Opcost").set(Max_Opcost)

  # Construct resources price dataframe, in order to send it to ampl
  resources=pd.DataFrame(
        [
            ("NatGasGrid", NatGasGrid),
            ("ElecGridBuy", ElecGridBuy),
            ("ElecGridSell", ElecGridSell),
            ("HydrogenGrid", HydrogenGrid),
        ],
        columns=["Grids", "c_spec"],
    ).set_index("Grids")
  
  # Set resources price
  ampl.set_data(resources, "Grids")

  # Construct heating level dataframe, in order to send it to ampl
  Theating=pd.DataFrame(
        [
            ("MediumT", 65),
            ("LowT", 50),
        ],
        columns=["HeatingLevel", "Theating"],
    ).set_index("HeatingLevel")

  # Set heating level
  ampl.set_data(Theating, "HeatingLevel")

  # Solve
  ampl.solve()

  # Save results in specified file
  save_ampl_results(ampl, pkl_name=result_file)

  # Return Results
  path_to_result="./codes_04_energy_system_integration/results/"+result_file+".pkl"
  myData = pd.read_pickle(path_to_result)
  return myData


class criteria1(Enum):
    OPEX = 'OPEX'
    CAPEX = 'CAPEX'
    Emissions = 'Emissions'
    TOTEX = 'TOTEX'
    parametric = 'parametric'

class criteria2(Enum):
    OPEX = 'OPEX'
    CAPEX = 'CAPEX'
    Emissions = 'Emissions'
    TOTEX = 'TOTEX'
    parametric = 'parametric'


def get_pareto(criteria1, criteria2,n=10,remove_tech_pareto=[],TAX_pareto=120e-6,Max_Emissions_pareto=1e30, Max_Totalcost_pareto=1e30, Max_Invcost_pareto=1e30,Max_Opcost_pareto=1e30,NatGasGrid_pareto=0.0303,ElecGridSell_pareto=-0.06,ElecGridBuy_pareto=0.0916,HydrogenGrid_pareto=0.3731, c_elec_pareto=75.3, c_gas_pareto=228):
  # Check that two different criterias are selected
  if criteria1==criteria2:
    print("You must select differents criterias for pareto optimization")
    return 0
 
  # Initialize the pareto data
  crit1_pareto_min_crit1=np.zeros(n)
  crit2_pareto_min_crit1=np.zeros(n)
  crit1_pareto_min_crit2=np.zeros(n)
  crit2_pareto_min_crit2=np.zeros(n)

  # Get bounds for optimization
  # Optimize unbounded on first criteria
  data_crit_1=optimize(criteria1,TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Invcost=Max_Invcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
  crit1_min=data_crit_1['object1']
  if criteria2==criteria2.Emissions:
    crit2_max=data_crit_1['Emissions'].values[0][0]
  elif criteria2==criteria2.OPEX:
    crit2_max=data_crit_1['OpCost'].values[0][0]
  elif criteria2==criteria2.CAPEX:
    crit2_max=data_crit_1['InvCost'].values[0][0]
  elif criteria2==criteria2.TOTEX:
    crit2_max=data_crit_1['Totalcost'].values[0][0]

  # Optimize unbounded on second criteria
  data_crit_2=optimize(criteria2,TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Invcost=Max_Invcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
  crit2_min=data_crit_2['object1']
  if criteria1==criteria1.Emissions:
    crit1_max=data_crit_2['Emissions'].values[0][0]
  elif criteria1==criteria1.OPEX:
    crit1_max=data_crit_2['OpCost'].values[0][0]
  elif criteria1==criteria1.CAPEX:
    crit1_max=data_crit_2['InvCost'].values[0][0]
  elif criteria1==criteria1.TOTEX:
    crit1_max=data_crit_2['Totalcost'].values[0][0]
  
  # Now that global bounds are known, let's discretize them in order to bound the optimization at different levels and retrieve the pareto curve
  crit1_span=np.linspace(crit1_min,crit1_max,n)
  crit2_span=np.linspace(crit2_min,crit2_max,n)

  # Main loop for pareto curve
  for j in range(0,n):
    # Get pareto curve by optimizing criteria 1 and constraining criteria 2
    if criteria2==criteria2.Emissions:
      data_pareto=optimize(criteria1,Max_Emissions=crit2_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Invcost=Max_Invcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit1_pareto_min_crit1[j]=data_pareto['object1']
      crit2_pareto_min_crit1[j]=data_pareto['Emissions'].values[0][0]
    elif criteria2==criteria2.OPEX:
      data_pareto=optimize(criteria1,Max_Opcost=crit2_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Invcost=Max_Invcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit1_pareto_min_crit1[j]=data_pareto['object1']
      crit2_pareto_min_crit1[j]=data_pareto['OpCost'].values[0][0]
    elif criteria2==criteria2.CAPEX:
      data_pareto=optimize(criteria1,Max_Invcost=crit2_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit1_pareto_min_crit1[j]=data_pareto['object1']
      crit2_pareto_min_crit1[j]=data_pareto['InvCost'].values[0][0]
    elif criteria2==criteria2.TOTEX:
      data_pareto=optimize(criteria1,Max_Totalcost=crit2_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Invcost=Max_Invcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit1_pareto_min_crit1[j]=data_pareto['object1']
      crit2_pareto_min_crit1[j]=data_pareto['Totalcost'].values[0][0]

    # Get pareto curve by optimizing criteria 2 and constraining criteria 1
    if criteria1==criteria1.Emissions:
      data_pareto=optimize(criteria2,Max_Emissions=crit1_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Invcost=Max_Invcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit2_pareto_min_crit2[j]=data_pareto['object1']
      crit1_pareto_min_crit2[j]=data_pareto['Emissions'].values[0][0]
    elif criteria1==criteria1.OPEX:
      data_pareto=optimize(criteria2,Max_Opcost=crit1_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Invcost=Max_Invcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit2_pareto_min_crit2[j]=data_pareto['object1']
      crit1_pareto_min_crit2[j]=data_pareto['OpCost'].values[0][0]
    elif criteria1==criteria1.CAPEX:
      data_pareto=optimize(criteria2,Max_Invcost=crit1_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Totalcost=Max_Totalcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit2_pareto_min_crit2[j]=data_pareto['object1']
      crit1_pareto_min_crit2[j]=data_pareto['InvCost'].values[0][0]
    elif criteria1==criteria1.TOTEX:
      data_pareto=optimize(criteria2,Max_Totalcost=crit1_span[j],TAX=TAX_pareto,remove_tech=remove_tech_pareto,Max_Emissions=Max_Emissions_pareto,Max_Invcost=Max_Invcost_pareto,Max_Opcost=Max_Opcost_pareto,NatGasGrid=NatGasGrid_pareto,ElecGridSell=ElecGridSell_pareto,ElecGridBuy=ElecGridBuy_pareto,HydrogenGrid=HydrogenGrid_pareto, c_elec=c_elec_pareto,c_gas=c_gas_pareto)
      crit2_pareto_min_crit2[j]=data_pareto['object1']
      crit1_pareto_min_crit2[j]=data_pareto['Totalcost'].values[0][0]
  
  # Put results in 2 dataframes. One for optimization on criteria 1, and the other for optiimization on criteria 2
  crit1_opt=pd.DataFrame(data={str(criteria1.value):crit1_pareto_min_crit1,str(criteria2.value):crit2_pareto_min_crit1})
  crit2_opt=pd.DataFrame(data={str(criteria1.value):crit1_pareto_min_crit2,str(criteria2.value):crit2_pareto_min_crit2})
  return crit1_opt,crit2_opt




def draw_pareto(dataframe1,dataframe2, criteria1_name='name_with_units', criteria2_name='name_with_units'):
  plt.title("Pareto Curve")
  plt.grid()
  plt.scatter(dataframe1[dataframe1.columns[1]].values,dataframe1[dataframe1.columns[0]].values,c='r',alpha=0.5)
  plt.scatter(dataframe2[dataframe2.columns[1]].values,dataframe2[dataframe2.columns[0]].values,c='b',alpha=0.5)
  plt.legend([criteria1_name+" optimization",criteria2_name+" optimization"])
  plt.xlabel(criteria2_name)
  plt.ylabel(criteria1_name)
  plt.show()




if __name__ == '__main__':

  ### Example 1 ####
  # I want to optimize the operational cost, and I know that NatGas will 0.32. Additionally, I do not want to use the technology SOFC, nor the R290_MT
  # What will be the Investment costs? and what technology will be used
  # Uncomment lines below:
  # data=optimize(criteria=criteria.OPEX,NatGasGrid=0.32,remove_tech=['SOFC','R290_MT'])
  print(data['use'][data['use']!=0].dropna())
  print(data['InvCost'].values[0][0])

  
  ### Example 2 ###
  # I want to draw the pareto front between TOTEX and Emissions
  # Uncomment lines below:
  # TOTEX,EMISSIONS=get_pareto(criteria1.TOTEX,criteria2.Emissions,n=14)
  # draw_pareto(TOTEX,EMISSIONS,"TOTEX [CHF/yr]", "Emissions [gCO2/yr]")

  # data=optimize(criteria=criteria.TOTEX)
  # print(data['use'][data['use']!=0].dropna())
  # print(data['InvCost'].values[0][0])
 




  # Old code, don't bother


  # from amplpy import AMPL, Environment
  # from functions import save_ampl_results
  # import pandas as pd
  # import os



  # ampl = AMPL()

  # ampl.cd("TA_files/1.Reference_case") # select the right folder

  # ampl.read("NLP_ref.mod")
  # ampl.readData("NLP_ref.dat")

  # ampl.setOption('solver', 'snopt')
  # ampl.setOption('presolve_eps', 8.53e-15)

  # data=pd.read_csv("./TA_files/data_MOES.csv")
  # data.index = ["Building" + str(i) for i in range(1,len(data)+1)] # the index of the dataframe has to match the values of the set "Buildings" in ampl

  # # Solve
  # ampl.solve()

  # save_ampl_results(ampl, pkl_name="scenario1")

  # myData = pd.read_pickle("results//scenario1.pkl")
  # myData.OPEX

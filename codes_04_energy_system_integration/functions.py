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


def optimize(criteria,TAX=120e-6,Max_Emissions=1e30, Max_Totalcost=1e30, Max_Invcost=1e30,Max_Opcost=1e30,NatGasGrid=0.0303,ElecGridSell=-0.06,ElecGridBuy=0.0916,HydrogenGrid=0.3731):
  
  ampl = AMPL()
  ampl.cd("./codes_04_energy_system_integration/ampl_files") 
  ampl.read("moes.mod")
  ampl.get_parameter("CO2tax").set(TAX)
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


  ampl.readData("moes.dat")

  ampl.read("moesSolar.mod")
  ampl.read("moesSolar.dat")
  ampl.read("moesSOFC.dat")
  ampl.read("moesHP_R290_LT.mod")
  ampl.read("moesHP_R290_LT.dat")
  ampl.read("moesHP_R290_MT.mod")
  ampl.read("moesHP_R290_MT.dat")
  ampl.read("moesHP_R1270_LT.mod")
  ampl.read("moesHP_R1270_LT.dat")
  ampl.read("moesHP_R1270_MT.mod")
  ampl.read("moesHP_R1270_MT.dat")
  ampl.read("moesboiler.dat")

  ampl.setOption('solver', 'cplex')
  ampl.setOption('presolve_eps', 5e-05)
  ampl.setOption('omit_zero_rows', 1)
  ampl.setOption('omit_zero_cols', 1)
  
  data=pd.read_csv("./codes_01_energy_demand/data_MOES.csv")
  data.index = ["Building" + str(i) for i in range(1,len(data)+1)] # the index of the dataframe has to match the values of the set "Buildings" in ampl

  # send parameters to ampl
  for col in data.columns:
    ampl.getParameter(col).setValues(data[col])
    
  ampl.get_parameter("Max_Emissions").set(Max_Emissions)
  ampl.get_parameter("Max_Totalcost").set(Max_Totalcost)
  ampl.get_parameter("Max_Invcost").set(Max_Invcost)
  ampl.get_parameter("Max_Opcost").set(Max_Opcost)
  #ampl.get_parameter("c_spec")
  resources=pd.DataFrame(
        [
            ("NatGasGrid", NatGasGrid),
            ("ElecGridBuy", ElecGridBuy),
            ("ElecGridSell", ElecGridSell),
            ("HydrogenGrid", HydrogenGrid),
        ],
        columns=["Grids", "c_spec"],
    ).set_index("Grids")
  ampl.set_data(resources, "Grids")
  ampl.setOption('solver', 'gurobi')
  ampl.solve()
  save_ampl_results(ampl, pkl_name="optimize_dump")
  myData = pd.read_pickle("./codes_04_energy_system_integration/results/optimize_dump.pkl")
  return myData


def solve_TOTEX(Max_Emissions=1e20):
  ampl = AMPL()
  ampl.cd("./codes_04_energy_system_integration/ampl_files") 
  ampl.read("moes.mod")
  ampl.read("objective_TOTEX.mod")
  ampl.readData("moes.dat")

  ampl.read("moesSolar.mod")
  ampl.readData("moesSolar.dat")
  ampl.read("moesSOFC.dat")
  ampl.read("moesHP_R290_LT.mod")
  ampl.read("moesHP_R290_LT.dat")
  ampl.read("moesHP_R290_MT.mod")
  ampl.read("moesHP_R290_MT.dat")
  ampl.read("moesHP_R1270_LT.mod")
  ampl.read("moesHP_R1270_LT.dat")
  ampl.read("moesHP_R1270_MT.mod")
  ampl.read("moesHP_R1270_MT.dat")
  ampl.read("moesboiler.dat")

  ampl.setOption('solver', 'cplex')
  ampl.setOption('presolve_eps', 5e-05)
  ampl.setOption('omit_zero_rows', 1)
  ampl.setOption('omit_zero_cols', 1)
  data=pd.read_csv("./codes_01_energy_demand/data_MOES.csv")
  data.index = ["Building" + str(i) for i in range(1,len(data)+1)] # the index of the dataframe has to match the values of the set "Buildings" in ampl

  # send parameters to ampl
  for col in data.columns:
    ampl.getParameter(col).setValues(data[col])
  ampl.get_parameter("Max_Emissions").set(Max_Emissions)

  ampl.setOption('solver', 'gurobi')
  ampl.solve()
  
  save_ampl_results(ampl, pkl_name="solve_TOTEX")
  myData = pd.read_pickle("./codes_04_energy_system_integration/results/solve_TOTEX.pkl")
  return myData['Totalcost'].values[0][0],myData['Emissions'].values[0][0]
  
def solve_Emissions(Max_Totalcost=1e20):
  ampl = AMPL()
  ampl.cd("./codes_04_energy_system_integration/ampl_files") 
  ampl.read("moes.mod")
  ampl.read("objective_Emissions.mod")
  ampl.readData("moes.dat")

  ampl.read("moesSolar.mod")
  ampl.readData("moesSolar.dat")
  ampl.read("moesSOFC.dat")
  ampl.read("moesHP_R290_LT.mod")
  ampl.read("moesHP_R290_LT.dat")
  ampl.read("moesHP_R290_MT.mod")
  ampl.read("moesHP_R290_MT.dat")
  ampl.read("moesHP_R1270_LT.mod")
  ampl.read("moesHP_R1270_LT.dat")
  ampl.read("moesHP_R1270_MT.mod")
  ampl.read("moesHP_R1270_MT.dat")
  ampl.read("moesboiler.dat")

  ampl.setOption('solver', 'cplex')
  ampl.setOption('presolve_eps', 5e-05)
  ampl.setOption('omit_zero_rows', 1)
  ampl.setOption('omit_zero_cols', 1)
  data=pd.read_csv("./codes_01_energy_demand/data_MOES.csv")
  data.index = ["Building" + str(i) for i in range(1,len(data)+1)] # the index of the dataframe has to match the values of the set "Buildings" in ampl

  # send parameters to ampl
  for col in data.columns:
    ampl.getParameter(col).setValues(data[col])
  ampl.get_parameter("Max_Totalcost").set(Max_Totalcost)

  ampl.setOption('solver', 'gurobi')
  ampl.solve()
  
  save_ampl_results(ampl, pkl_name="solve_Emissions")
  myData = pd.read_pickle("./codes_04_energy_system_integration/results/solve_Emissions.pkl")
  return myData['Totalcost'].values[0][0],myData['Emissions'].values[0][0]

def draw_pareto(n):
  Emissions_pareto_min_TOTEX=np.zeros(n)
  Totalcost_pareto_min_TOTEX=np.zeros(n)
  Emissions_pareto_min_Emissions=np.zeros(n)
  Totalcost_pareto_min_Emissions=np.zeros(n)
  Totalcost_max,Emissions_min=solve_Emissions(1e20)
  Totalcost_min,Emissions_max=solve_TOTEX(1e20)
  
  Emissions_span=np.linspace(Emissions_min,Emissions_max,n)
  Totalcost_span=np.linspace(Totalcost_min,Totalcost_max,n)
  for j in range(0,n):
    Totalcost_pareto_min_TOTEX[j],Emissions_pareto_min_TOTEX[j]=solve_TOTEX(Emissions_span[j])
    Totalcost_pareto_min_Emissions[j],Emissions_pareto_min_Emissions[j]=solve_Emissions(Totalcost_span[j])
  return Totalcost_pareto_min_TOTEX,Emissions_pareto_min_TOTEX,Totalcost_pareto_min_Emissions,Emissions_pareto_min_Emissions









if __name__ == '__main__':

  data=optimize(criteria.Emissions, TAX=400e-6)
  print(data['c_elec'])
  print(data['Emissions'])
  print(data['OpCost'])
  print(data['mult'])
  print(data['CO2tax'])
  # Totcost_TOTEX,Emiss_TOTEX,Totcost_Emissions,Emiss_Emissions=draw_pareto(20)
  # plt.scatter(Totcost_TOTEX,Emiss_TOTEX,c='r',alpha=0.3)
  # plt.scatter(Totcost_Emissions,Emiss_Emissions,c='b',alpha=0.3)
  # plt.legend(["TOTEX optimization","Emissions optimization"])
  # plt.grid()
  # plt.show()
  # print(Emiss_Emissions)
  # print(Totcost_Emissions)
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

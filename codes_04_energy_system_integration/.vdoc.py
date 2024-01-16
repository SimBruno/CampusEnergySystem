# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from amplpy import AMPL, Environment
ampl = AMPL()

#ampl.cd("C:/Users/coren/Desktop/report-group-3/codes_04_energy_system_integration/ampl_files") # select the right folder
ampl.cd("./codes_04_energy_system_integration/ampl_files") 

ampl.read("moes.mod")
ampl.read("objective_TOTEX.mod")
ampl.readData("moes.dat")

ampl.read("moesSolar.mod")
ampl.readData("moesSolar.dat")

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

#
#
#
#
#
#
import pandas as pd
import os

data=pd.read_csv("./codes_01_energy_demand/data_MOES.csv")
data.index = ["Building" + str(i) for i in range(1,len(data)+1)] # the index of the dataframe has to match the values of the set "Buildings" in ampl

# send parameters to ampl
 for col in data.columns:
  ampl.getParameter(col).setValues(data[col])

data

Max_Emissions=1e20
Emissions_global_min=6859987172
Emissions_global_max=9.200615e+09
ampl.get_parameter("Max_Emissions").set(Max_Emissions)
#
#
#
#
#
# Solve
ampl.setOption('solver', 'gurobi')
ampl.solve()
#
#
#
#
#
#
import sys
#sys.path.append('C:/Users/coren/Desktop/report-group-3/codes_04_energy_system_integration/')
sys.path.append('./codes_04_energy_system_integration/')
from functions import save_ampl_results
save_ampl_results(ampl, pkl_name="scenario1")
#
#
#
#
#
import pandas as pd
#myData = pd.read_pickle("C:/Users/coren/Desktop/report-group-3/codes_04_energy_system_integration/results/scenario1.pkl")
myData = pd.read_pickle("./codes_04_energy_system_integration/results/scenario1.pkl")
myData["Totalcost"]
myData["k_th"]
myData["Emissions"]
myData["use"]
myData["mult_t"].loc["ElecGridBuy"]
myData["mult_t"].loc["NatGasGrid"]
myData["mult_t"].loc["Boiler"]
myData["mult_t"].loc["STC"]
```
#
#
#
#
import codes_04_energy_system_integration.functions as fct3
import pandas as pd
data=fct3.optimize(criteria=criteria.OPEX,NatGasGrid=0.32)
#
#
#
print("The technologies used are:")
print(data['use'][data['use']!=0].dropna().index.values)

print("The investment cost is %.3e [CHF/yr]" %(data['InvCost'].values[0][0]))
#
#
#
#
#
#
import codes_04_energy_system_integration.functions as fct3
import pandas as pd

TOTEX,EMISSIONS=fct3.get_pareto(criteria1.TOTEX,criteria2.Emissions,n=30)
#
#
#
import codes_04_energy_system_integration.functions as fct3
import pandas as pd

fct3.draw_pareto(TOTEX,EMISSIONS,"TOTEX [CHF/yr]", "Emissions [gCO2/yr]")
#
#
#
#
#

data=fct3.optimize(criteria=criteria.CAPEX)
#
#
#
print("The technologies used are:")
print(data['use'][data['use']!=0].dropna().index.values)
print(data['mult_t'].loc["Boiler"])
print(data['mult_t'].loc["ElecGridBuy"])
print(data['mult_t'].loc["NatGasGrid"])
print("The investment cost is %.3e [CHF/yr]" %(data['InvCost'].values[0][0]))
print("The operating cost is %.3e [CHF/yr]" %(data['OpCost'].values[0][0]))
print("The CO2 emissions are %.3e [g/yr]" %(data['Emissions'].values[0][0]))
print("The total cost is %.3e [CHF/yr]" %(data['Totalcost'].values[0][0]))
#
#
#
#
#

data=fct3.optimize(criteria=criteria.OPEX)
#
#
#
print("The technologies used are:")
print(data['use'][data['use']!=0].dropna().index.values)
print(data['mult_t'].loc["Boiler"])
print(data['mult_t'].loc["R1270_LT"])
print(data['mult_t'].loc["PV"])
print(data['mult_t'].loc["ElecGridBuy"])
print(data['mult_t'].loc["NatGasGrid"])
print("The investment cost is %.3e [CHF/yr]" %(data['InvCost'].values[0][0]))
print("The operating cost is %.3e [CHF/yr]" %(data['OpCost'].values[0][0]))
print("The CO2 emissions are %.3e [g/yr]" %(data['Emissions'].values[0][0]))
print("The total cost is %.3e [CHF/yr]" %(data['Totalcost'].values[0][0]))
#
#
#
#

data=fct3.optimize(criteria=criteria.Emissions)
#
#
#
print("The technologies used are:")
print(data['use'][data['use']!=0].dropna().index.values)
print(data['mult_t'].loc["STC"])
print(data['mult_t'].loc["R290_MT"])
print(data['mult_t'].loc["SOFC"])
print(data['mult_t'].loc["HydrogenGrid"])
print(data['mult_t'].loc["ElecGridBuy"])
print(data['mult_t'].loc["ElecGridSell"])
print("The investment cost is %.3e [CHF/yr]" %(data['InvCost'].values[0][0]))
print("The operating cost is %.3e [CHF/yr]" %(data['OpCost'].values[0][0]))
print("The CO2 emissions are %.3e [g/yr]" %(data['Emissions'].values[0][0]))
print("The total cost is %.3e [CHF/yr]" %(data['Totalcost'].values[0][0]))
#
#
#
#

data=fct3.optimize(criteria=criteria.TOTEX)
#
#
#
print("The technologies used are:")
print(data['use'][data['use']!=0].dropna().index.values)
print(data['mult_t'].loc["STC"])
print(data['mult_t'].loc["Boiler"])
print(data['mult_t'].loc["NatGasGrid"])
print(data['mult_t'].loc["ElecGridBuy"])
print("The investment cost is %.3e [CHF/yr]" %(data['InvCost'].values[0][0]))
print("The operating cost is %.3e [CHF/yr]" %(data['OpCost'].values[0][0]))
print("The CO2 emissions are %.3e [g/yr]" %(data['Emissions'].values[0][0]))
print("The total cost is %.3e [CHF/yr]" %(data['Totalcost'].values[0][0]))
#
#
#
#

data=fct3.optimize(criteria=criteria.parametric)
#
#
#
#
print("The technologies used are:")
print(data['use'][data['use']!=0].dropna().index.values)
print(data['mult_t'].loc["STC"])
print(data['mult_t'].loc["Boiler"])
print(data['mult_t'].loc["NatGasGrid"])
print(data['mult_t'].loc["ElecGridBuy"])
print("The investment cost is %.3e [CHF/yr]" %(data['InvCost'].values[0][0]))
print("The operating cost is %.3e [CHF/yr]" %(data['OpCost'].values[0][0]))
print("The CO2 emissions are %.3e [g/yr]" %(data['Emissions'].values[0][0]))
print("The total cost is %.3e [CHF/yr]" %(data['Totalcost'].values[0][0])) 
#
#
#
#
#
#
#
#
#
#

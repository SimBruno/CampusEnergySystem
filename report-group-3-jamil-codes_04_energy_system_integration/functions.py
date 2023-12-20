import pickle

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
  #result_file_path = "results//"+ pkl_name+ ".pkl"
  #result_file_path = "C:/Users/coren/Desktop/report-group-3/codes_04_energy_system_integration/results/scenario1.pkl"
  result_file_path = "C:/Users/coren/Desktop/report-group-3/report-group-3-jamil-codes_04_energy_system_integration/results/scenario1.pkl"
  f = open(result_file_path, 'wb')
  pickle.dump(results, f)
  f.close()
  return 



if __name__ == '__main__':


  from amplpy import AMPL, Environment
  from functions import save_ampl_results
  import pandas as pd
  import os

  ampl = AMPL()

  ampl.cd("TA_files/1.Reference_case") # select the right folder

  ampl.read("NLP_ref.mod")
  ampl.readData("NLP_ref.dat")

  ampl.setOption('solver', 'snopt')
  ampl.setOption('presolve_eps', 8.53e-15)

  data=pd.read_csv("./TA_files/data_MOES.csv")
  data.index = ["Building" + str(i) for i in range(1,len(data)+1)] # the index of the dataframe has to match the values of the set "Buildings" in ampl

  # Solve
  ampl.solve()

  save_ampl_results(ampl, pkl_name="scenario1")

  myData = pd.read_pickle("results//scenario1.pkl")
  myData.OPEX

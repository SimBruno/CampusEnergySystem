from amplpy import AMPL
import pandas as pd
import os
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
    #result_file_path = "./codes_04_energy_system_integration/results/"+ pkl_name + ".pkl"
    result_file_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ),'results',pkl_name+".pkl"))
    print(result_file_path)
    #result_file_path = "./codes_04_energy_system_integration/results/scenario1.pkl"
    f = open(result_file_path, 'wb')
    pickle.dump(results, f)
    f.close()
    return 

def run_ampl(data_file, model_directory, model_file, buildings_required=False):

    # Read data from CSV
    # data_file = "clusters_data.csv"
    data_path = os.path.join(".","codes_01_energy_demand", data_file)
    data = pd.read_csv(data_path)

    # model_file = "NLP_ref.mod"
    # model_directory = "1.Reference_case_Mirco"
    model_path = os.path.join(os.getcwd(),"codes_02_heat_recovery",model_directory, model_file)

    # Create an AMPL instance
    ampl = AMPL()

    # Load the model
    ampl.read(model_path)

    # Pass data to AMPL
    ampl.set['Time']=set(data.index)
    ampl.getParameter("top").setValues(data['Hours'])

    # if buildings properties are needed, load them
    if buildings_required==True:
        ampl.getParameter("Text").setValues(data['Temp'])
        ampl.getParameter("irradiation").setValues(data['Irr']/1000)

        buildings_file  = "thermal_properties.csv"
        buildings_path  = os.path.join(os.getcwd(),"codes_01_energy_demand", buildings_file)
        buildings       = pd.read_csv(buildings_path)
        buildings.index = [f'Building{i}' for i in range(1,len(buildings)+1)]

        #drop specQ_people col
        buildings.drop(columns= "specQ_people", inplace=True)

        for col in buildings.columns:
            ampl.getParameter(col).setValues(buildings[col])
      
    else: 
        ampl.getParameter("Qheating").setValues(data['Q_th'])


    # Solve the model
    ampl.setOption('solver', 'snopt')
    #ampl.setOption('presolve_eps', 8.53e-15)
    #ampl.setOption('omit_zero_rows', 1)
    #ampl.setOption('omit_zero_cols', 1)
    ampl.solve()
    #assert ampl.solve_result == "solved"

    results_name = model_file[:-4]
    print(results_name)

    save_ampl_results(ampl, pkl_name=results_name)

    print("Optimization done\n")

    return

if __name__ == "__main__":
    # Code to be executed when the script is run directly

    # Read data from CSV
    data_file = "clusters_data.csv"
    model_file = "NLP_vent.mod"
    model_directory = "3.Ventilation"

    run_ampl(data_file, model_directory, model_file,True)

    print("Script executed successfully.")

    data = pd.read_pickle(r'C:\Users\maj\Desktop\report-group-3\codes_02_heat_recovery\results\NLP_vent.pkl')
    print(data['OPEX'])

    print(data['CAPEX'])
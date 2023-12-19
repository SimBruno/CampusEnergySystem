from amplpy import AMPL
import pandas as pd
import os

def run_ampl(data_file, model_directory, model_file):

    # Read data from CSV
    # data_file = "clusters_data.csv"
    data_path = os.path.join(os.getcwd(),"codes_01_energy_demand", data_file)
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
    ampl.getParameter("Qheating").setValues(data['Q_th'])

    # Solve the model
    ampl.solve()

    totalcost = ampl.getObjective("Totopex")
    #print totalcost
    print(totalcost.getValues())


import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os


# Parameters
# NEEDS TO BE COMPLETED
# Pay attention to the physical units implied !

h = 8760  # Hours in a year
T_th = 273 + 16 # Cut off temperature [K]
cp_air = 1152 # Air specific heat capacity [J/(m3.K)] 
T_int = 273 + 21 # Set point temperature [K]
air_new = 2.5 # Air renewal [m3/(m2.h)]
Vent = 0 # [...]
f_el = 0.8 # Share of electricity demand which is converted to heat appliances



def load_data_weather_buildings():
    
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py

    weather = pd.read_csv(os.path.join(path, "Weather.csv"),header=0,encoding = 'unicode_escape')
    weather.columns = ['Temp', 'Irr']

    buildings = pd.read_csv(os.path.join(path, "Buildings.csv"),header=0,encoding = 'unicode_escape')
    buildings.columns = ['Name', 'Year', 'Ground', 'Heat', 'Elec']

    return weather, buildings


def occupancy_profile():
    # Define occupancy and electricity profiles
    occ_off = [0, 0, 0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.4, 0.6, 0.8, 0.8, 0.4, 0.2, 0, 0, 0, 0, 0, 0]
    occ_class = [0, 0, 0, 0, 0, 0, 0, 0.4, 0.6, 1, 1, 0.8, 0.2, 0.6, 1, 0.8, 0.8, 0.4, 0, 0, 0, 0, 0, 0]
    occ_can = [0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.2, 0.4, 1, 0.4, 0.2, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    weekday_elec = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    occ_all = pd.DataFrame({'Office': occ_off, 'Class': occ_class, 'Restaurant': occ_can, 'Electricity': weekday_elec})

    # Create a date range
    start_date = '2023-01-01'
    end_date = '2024-01-01' 
    date_range = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='right')

    occupancy = occ_all # initialize
    while len(occupancy) < len(date_range): #extend to a year
        occupancy= pd.concat([occupancy, occ_all], ignore_index=True)
    occupancy = occupancy.iloc[0:len(date_range),:]  # drop extra rows if needed

    # Create the occupancy DataFrame
    occupancy.loc[(date_range.weekday >= 5), ['Office', 'Class', 'Restaurant', 'Electricity']] = 0 #set to 0 on weekends
    return occupancy.Office.values, occupancy.Class.values, occupancy.Restaurant.values, occupancy.Electricity.values


def people_gains(building_id, profile_class, profile_rest, profile_off):
    # Heat gains from people (Office, Restaurant, Classroom)
    hg_off=5
    hg_rest=35
    hg_class=23.3
    hg_others=0
    
    # Share areas (Office, Restaurant, Classroom)
    A_off=0.3
    A_rest=0.05
    A_class=0.35
    A_others=0.3

    # Scaling factors

    sf_off = hg_off*A_off
    sf_rest = hg_rest*A_rest
    sf_class = hg_class*A_class
    sf_others = hg_others*A_others
    return sf_off*profile_off + sf_rest*profile_rest + sf_class*profile_class


def elec_gains(building_id, profile_elec):
    #elec_build=buildings.Elec ###Wh
    cf = 1000/profile_elec.sum() # Conversion factor from Wh to W and capacity factor
    #A_th=buildings['Ground'].loc[buildings.Name==building_id].values[0] # m2
    #E_elec = buildings['Elec'].loc[buildings.Name==building_id].values[0] # W
    #q_elec_hour = E_elec*cf/A_th # W/m2, for one hour when on -> uniform distribution assumption
    #q_elec_hourly=np.zeros(len(profile_elec)) # create q_elec(t) over the year
    #q_elec_hourly[profile_elec==1]=q_elec_hour # fill q_elec(t) with q_elec_hour when on
    return cf*profile_elec*buildings.loc[buildings.Name==building_id].apply(lambda x: x['Elec']/x['Ground'] , axis=1).to_numpy()[0] # W/m2 for each hour of the year


"""def solving_NR(tolerance,max_iteration,building_id, k_th_guess, k_sun_guess):
    
    # Initialize counters and tolerances
    e_th = 1
    e_sun = 1
    iteration=0

    # Initialize guess values
    k_th = k_th_guess
    k_sun = k_sun_guess

    # Getting other parameters
    A_th=buildings['Ground'].loc[buildings.Name==building_id].values[0] # m2
    Q_th=buildings['Heat'].loc[buildings.Name==building_id].values[0]*1000 # W
    T_ext=weather.Temp + 273 # K
    irr=weather.Irr # W/m2

    #Compute mean values for irradiance, heat gain from people and appliances
    cutoff_indicator = ((T_th -1 <= T_ext) & (T_ext <= T_th + 1)) # state switch on/off conditions
    #cutoff_indicator = ((T_ext >= T_th +1) & (T_ext <= T_th - 1)) # state switch on/off conditions

    q_elec_mean = q_elec[cutoff_indicator].mean() 
    q_people_mean = q_people[cutoff_indicator].mean()
    irr_mean = irr[cutoff_indicator].mean()

    specQ_people = q_people.mean()
    # Newton Raphson method
    
    alpha = 0.5 # relaxation factor

    while iteration < max_iteration:

        # Build f(k_th,k_sun) = 0
        h = A_th* (k_th*(T_int-T_ext)- q_people - k_sun*irr - q_elec*f_el) #auxiliary function to determine the heating mode

        heating_indicator = ((T_ext <= T_th) & (h>=0)) # state heating mode conditions

        f = h[heating_indicator].sum() - Q_th # filter out negative values of hx and sum the remaining values

        # Build g(k_th,k_sun) = 0

        g = A_th* (k_th*(T_int-T_th) - k_sun*irr_mean - q_people_mean - q_elec_mean*f_el)
   
        # Build Jacobian
        J = A_th*np.array([ [(T_int - T_ext[heating_indicator]).sum(),  -irr[heating_indicator].sum()],
                            [(T_int - T_th),                              -irr_mean]])
        
        # Calculate the update step
        delta = [f, g]
        # Solve the linear system using matrix inversion
        delta_k_th, delta_k_sun = np.linalg.solve(J, delta)*(1-alpha)

        k_th_old = k_th
        k_sun_old = k_sun

        # Update k_th and k_sun
        k_th -= delta_k_th
        k_sun -= delta_k_sun

        # Calculate the relative errors
        e_th = abs(f)
        e_sun = abs(g)
        
        if (e_th < tolerance) and (e_sun < tolerance) and (abs(k_th - k_th_old) < tolerance*(1+ abs(k_th_old))) and (abs(k_sun - k_sun_old) < tolerance*(1+ abs(k_sun_old))):
            break  # Converged, exit the loop
        iteration += 1        
    print(building_id, iteration, k_th, k_sun, e_th, e_sun)

    return k_th, k_sun, iteration, e_th, e_sun, A_th, specQ_people, q_elec_mean 
"""

def solving_NR(tolerance,max_iteration,building_id, k_th_guess, k_sun_guess):
    
    # Initialize counters and tolerances
    e_th = 1
    e_sun = 1
    iteration=0

    # Initialize guess values
    k_th = k_th_guess
    k_sun = k_sun_guess

    # Extract other parameters
    A_th=buildings['Ground'].loc[buildings.Name==building_id].values[0] # m2
    Q_th=buildings['Heat'].loc[buildings.Name==building_id].values[0]*1000 # W
    T_ext=weather.Temp + 273 # K
    irr=weather.Irr # W/m2

    #Compute mean values for irradiance, heat gain from people and appliances
    cutoff_indicator = ((T_th -1 <= T_ext) & (T_ext <= T_th + 1) & (profile_elec==1)) # state switch on/off conditions

    #q_elec_mean = q_elec[cutoff_indicator].mean() 
    #q_people_mean = q_people[cutoff_indicator].mean()
    #irr_mean = irr[cutoff_indicator].mean()

    q_elec_mean = q_elec.mean() 
    q_people_mean = q_people.mean()
    irr_mean = irr.mean()
    specQ_people = q_people.mean()
    # Newton Raphson method
    
    #alpha = 0.7 # relaxation factor

    while iteration < max_iteration:

        # Build f(k_th,k_sun) = 0
        h =(k_th*(T_int-T_ext) - q_people - k_sun*irr - q_elec*f_el) #auxiliary function to determine the heating mode

        heating_indicator = ((T_ext <= T_th) & (h>=0) & (profile_class>0)) # state heating mode conditions

        f = h[heating_indicator].sum() - Q_th/A_th # filter out negative values of hx and sum the remaining values
        # Build derivative
        dfdx =(T_int - T_ext[heating_indicator]).sum()
        
        k_th_old = k_th
        k_sun_old = k_sun    

        # Solve the linear system using matrix inversion
        k_th -= f/dfdx

        # Update k_sun
        k_sun = (k_th*(T_int-T_th) - q_people_mean - q_elec_mean*f_el)/irr_mean

        g =(k_th*(T_int-T_th) - k_sun*irr_mean - q_people_mean - q_elec_mean*f_el)

        # Calculate the relative errors
        e_th = abs(f)
        e_sun = abs(g)
        
        if (e_th < tolerance) and (e_sun < tolerance) and (abs(k_th - k_th_old) < tolerance*(1+ abs(k_th_old))) and (abs(k_sun - k_sun_old) < tolerance*(1+ abs(k_sun_old))):
            break  # Converged, exit the loop
        iteration += 1        
    #print(building_id, iteration, k_th, k_sun, e_th, e_sun)

    return k_th, k_sun, iteration, e_th, e_sun, A_th, specQ_people, q_elec_mean 



if __name__ == '__main__': 
    # the code below will be executed only if you run the NR_function.py file as main file, not if you import the functions from another file (another .py or .qmd)
    
    # Load data
    weather, buildings = load_data_weather_buildings()

    #Compute gains and profile
    profile_off, profile_class, profile_rest, profile_elec = occupancy_profile()

    # State required tolerances and maximum number of iterations
    tol=1e-6
    max_iteration=10000

    # State initial guesses for k_th and k_sun
    k_th_guess=5
    k_sun_guess=.1

    # Initialize array to record values for each building
    
    Solution  = pd.DataFrame(columns=['FloorArea', 'specElec', 'k_th', 'k_sun', 'specQ_people'])

    for building_id in buildings['Name']:
        q_people = people_gains(building_id, profile_class, profile_rest, profile_off)
        q_elec = elec_gains(building_id, profile_elec)
        [k_th, k_sun, number_iteration, error1,error2, A_th, specQ_people, q_elec_mean] = solving_NR(tol,max_iteration,building_id,k_th_guess,k_sun_guess)
        Solution.loc[building_id] = pd.Series({'FloorArea': A_th, 'specElec': q_elec_mean, 'k_th': k_th, 'k_sun': k_sun,'specQ_people': specQ_people})
    
    #Saving dataframe in thermal_properties.csv
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    Solution.to_csv(os.path.join(path, "thermal_properties.csv"),index=False)

    #Printing solutions
    print(Solution)
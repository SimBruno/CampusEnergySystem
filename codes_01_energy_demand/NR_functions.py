
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
T_th = 288 # Cut off temperature [K]
cp_air = 1152 # Air specific heat capacity [J/(m3.K)] 
T_int = 294 # Set point temperature [K]
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
    # NEEDS TO BE COMPLETED
    # Daily weekday profile for office, canteen and classroom
    occ_off=[0,0,0,0,0,0,0,0.2,0.4,0.6,0.8,0.8,0.4,0.6,0.8,0.8,0.4,0.2,0,0,0,0,0,0]
    occ_class=[0,0,0,0,0,0,0,0.4,0.6,1,1,0.8,0.2,0.6,1,0.8,0.8,0.4,0,0,0,0,0,0]
    occ_can=[0,0,0,0,0,0,0,0,0.4,0.2,0.4,1,0.4,0.2,0.4,0,0,0,0,0,0,0,0,0]
    weekend=[0]*24
    week_off=occ_off*5 + weekend + weekend
    week_class=occ_class*5 + weekend + weekend
    week_can=occ_can*5 + weekend + weekend
    weekday_elec=[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
    week_elec=weekday_elec*5 + weekend + weekend
    yearly_off=week_off*52 + occ_off  ##### Lundi *2
    yearly_class=week_class*52 + occ_class
    yearly_can=week_can*52 + occ_can
    yearly_elec=week_elec*52 + weekday_elec
    # Yearly profile considering weekends for each usage (office, canteen and classroom)
      
    return [yearly_off,yearly_class,yearly_can,yearly_elec]


def people_gains(building_id: str, occ_profile):
    
    # Heat gains from people (Office, Restaurant, Classroom)
    heat_gain_off=5
    heat_gain_rest=35
    heat_gain_class=23.3
    heat_gain_others=0
    
    # Share areas (Office, Restaurant, Classroom)
    share_off=0.3
    share_rest=0.05
    share_class=0.35
    share_others=0.33
    
    # Yearly profile of heat gains from people
    surface_building_value=float(buildings[buildings['Name']==building_id]['Ground'])
    Q_build_hourly=[0]*len(occ_profile[1])
    Q_build_tot=[0]*len(occ_profile[1])
    for i in range(len(occ_profile[1])):
        Q_build_hourly[i]=heat_gain_off*share_off*occ_profile[0][i] + heat_gain_rest*share_rest*occ_profile[2][i] + heat_gain_class*share_class*occ_profile[1][i]
        Q_build_tot[i]=Q_build_hourly[i]*surface_building_value ### W hourly
    return Q_build_tot 



def elec_gains(building_id: str, occ_profile):
    #elec_build=buildings.Elec ###Wh
    elec_build_value=float(buildings[buildings['Name']==building_id]['Elec'])
    elec_hour=elec_build_value/3654 ##### W
    elec_gain=[0]*len(occ_profile[1])
    for i in range(len(occ_profile[1])):
        if occ_profile[3][i]==1:
            elec_gain[i]=elec_hour
    return elec_gain





def solving_NR(tolerance,max_iteration,building_id: str,k_th_guess,k_sun_guess):
    
    # Initialize counters and tolerances
    error=[1,1]
    counter=0

    # Initialize guess values
    k=np.array([k_th_guess,k_sun_guess])

    # Mean values calculation (for second part of the function)
    irr_mean=weather['Irr'].mean()
    Q_build_tot_mean=np.mean(Q_build_tot)
    elec_gain_mean=np.mean(elec_gain)

    # Getting other parameters
    surface_building_value=float(buildings[buildings['Name']==building_id]['Ground'])
    buildings_annual_heat=buildings.Heat
    buildings_annual_heat_value=1000*(float(buildings[buildings['Name']==building_id]['Heat']))
    temperature=weather.Temp
    irradiance=weather.Irr

    # Construct problem
    while (((np.abs(error[0])>=tolerance) or (np.abs(error[1])>=tolerance)) and (counter<max_iteration)): #Note that there are 2 errors to check. This is because the function is bidimmensional

        # func1 is a two dimmensional function, grouping equations 1 and 2 of the problem
        func1=np.array([0,surface_building_value*(k[0]*(T_int-T_th)-k[1]*irr_mean)-Q_build_tot_mean-f_el*elec_gain_mean])

        #Jacobian is the derivative of func1, wrt k_th and k_sun
        jacobian=np.array([[0,0],[surface_building_value*(T_int-T_th),-surface_building_value*irr_mean]])

        #Construction of the first part of func1
        for j in range(len(Q_build_tot)):
            func1_temp=surface_building_value*(k[0]*(T_int-273-temperature[j])-k[1]*irradiance[j])-Q_build_tot[j]-f_el*elec_gain[j]
            if ((func1_temp>=0) and (temperature[j]+273<=T_th)): #Add value only if on "Heating mode"
                func1[0]=func1[0]+func1_temp
                jacobian[0,0]=jacobian[0,0]+surface_building_value*(T_int-temperature[j]-273)
                jacobian[0,1]=jacobian[0,1]-surface_building_value*irradiance[j]
        
        #Lastly, since the newton raphson model seeks solution for F(x)=0, we must substract the annual heating value to the first part of the function
        func1[0]=func1[0]-buildings_annual_heat_value

        # Solving with the Newton Raphson method
        k=k.transpose()-np.dot(np.linalg.inv(jacobian),func1)
    

        # Compute error
        func1=np.array([0,surface_building_value*(k[0]*(T_int-T_th)-k[1]*irr_mean)-Q_build_tot_mean-f_el*elec_gain_mean])
        for j in range(len(Q_build_tot)):
            func1_temp=surface_building_value*(k[0]*(T_int-273-temperature[j])-k[1]*irradiance[j])-Q_build_tot[j]-f_el*elec_gain[j]
            if ((func1_temp>=0) and (temperature[j]+273<=T_th)):
                func1[0]=func1[0]+func1_temp
        error=np.array([func1[0]-buildings_annual_heat_value,func1[1]])
        counter=counter+1



    #############Old guidelines###################

    # define the conditions for switching on the heating system
    ## T ext < T cutoff
    ## During opening hours of the campus
    ...
    
    # find the mean values for irradiance, heat gain from people and appliances over the +/- 1°C interval around the T cutoff (for the second equation)
    ...
    
    # initialise NR
    ...
    
    # iterate over k and solve the NR 2-D equations. Don't forget to consider only positive heat demands (non linear term of the first equation) 
    # Check the termination criteria (epsilon and max iter)
    ...
    
    return k,counter,error



if __name__ == '__main__': 
    # the code below will be executed only if you run the NR_function.py file as main file, not if you import the functions from another file (another .py or .qmd)
    

    # Load data
    weather, buildings = load_data_weather_buildings()

    #Compute gains and profile
    occ_profile = occupancy_profile()
    

    # State required tolerances and maximum number of iterations
    tolerance=0.00001
    max_iteration=10000

    # State initial guesses for k_th and k_sun
    k_th_guess=1
    k_sun_guess=0.1

    # Initialize array to record values for each building
    k_th=[0]*len(buildings)
    k_sun=[0]*len(buildings)
    number_iteration=[0]*len(buildings)
    error1=[0]*len(buildings)
    error2=[0]*len(buildings)

    # Loop to get values for each building
    count=0
    for building_id in buildings['Name']:
        Q_build_tot = people_gains(building_id, occ_profile)
        elec_gain=elec_gains(building_id, occ_profile)
        [[k_th[count],k_sun[count]],number_iteration[count],[error1[count],error2[count]]]=solving_NR(tolerance,max_iteration,building_id,k_th_guess,k_sun_guess)
        count=count+1
  
    #Storing everything in a pandas dataframe
    data={'Name':buildings['Name'].to_numpy(), 'k_th': k_th, 'k_sun':k_sun,'number_iteration':number_iteration,'error1':error1,'error2':error2} #Note that there are 2 errors. This is because the function is bidimmensional
    Solution=pd.DataFrame(data)

    #Saving dataframe in thermal_properties.csv
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    Solution.to_csv(os.path.join(path, "thermal_properties.csv"),index=False)

    #Printing solutions
    print(Solution)

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
T_th = 289 # Cut off temperature [K]
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


def people_gains(building_id, occ_profile):
    
    # NEEDS TO BE COMPLETED
   
    
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
    result=occupancy_profile()
    surface_building=buildings.Ground
    surface_building_value=surface_building.iloc[building_id]
    Q_build_hourly=[0]*len(result[1])
    Q_build_tot=[0]*len(result[1])
    for i in range(len(result[1])):
        Q_build_hourly[i]=heat_gain_off*share_off*result[0][i] + heat_gain_rest*share_rest*result[2][i] + heat_gain_class*share_class*result[1][i]
        Q_build_tot[i]=Q_build_hourly[i]*surface_building_value ### W hourly
    return Q_build_tot 



def elec_gains(building_id, occ_profile):
    elec_build=buildings.Elec ###Wh
    elec_build_value=elec_build.iloc[building_id]
    result=occupancy_profile()
    elec_hour=elec_build_value/3654 ##### W
    elec_gain=[0]*len(result[1])
    for i in range(len(result[1])):
        if result[3][i]==1:
            elec_gain[i]=elec_hour
   
    return elec_gain





def solving_NR():
    
    # NEEDS TO BE COMPLETED
    
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
    
    
    return ...



if __name__ == '__main__': 
    # the code below will be executed only if you run the NR_function.py file as main file, not if you import the functions from another file (another .py or .qmd)
    
    weather, buildings = load_data_weather_buildings()
    print(buildings)
    
    occ_profile = occupancy_profile()

    building_id=1
    #occ_profile=None
    people_gains(building_id, occ_profile)

    building_id=12
    elec_gains(building_id, occ_profile)


    solving_NR()

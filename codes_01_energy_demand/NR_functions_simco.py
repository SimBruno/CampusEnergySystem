
import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os


# Parameters
# NEEDS TO BE COMPLETED
# Pay attention to the physical units implied !

h = ... # Hours in a year
T_th = ... # Cut off temperature [...]
cp_air = ... # Air specific heat capacity [...] 
T_int = ... # Set point temperature [...]
air_new = ... # Air renewal [...]
Vent = ... # [...]
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
    ...
    
    # Yearly profile considering weekends for each usage (office, canteen and classroom)
    ...  
    
    return ...


def people_gains(building_id, occ_profile):
    
    # NEEDS TO BE COMPLETED

    # Heat gains from people (Office, Restaurant, Classroom)
    ...
    
    # Share areas (Office, Restaurant, Classroom)
    ...
    
    # Yearly profile of heat gains from people
    ...
    
    return ...



def elec_gains(building_id, occ_profile):
    
    # NEEDS TO BE COMPLETED
    #Goal: from annual demand to hourly demand because we are interested in annual demand of the buidling to get hours by hours profile
    
    buildings.Elec
    
    return ...









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

    building_id=None
    occ_profile=None
    people_gains(building_id, occ_profile)

    building_id= 1
    elec_gains(building_id, occ_profile)


    solving_NR()

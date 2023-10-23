
import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os



####################
#### Parameters ####
####################

h = 8760  # Hours in a year
T_th = 289 # Cut off temperature [K]
cp_air = 1152 # Air specific heat capacity [J/(m3.K)] 
T_int = 294 # Set point temperature [K]
air_new = 2.5 # Air renewal [m3/(m2.h)]
Vent = 0 # [...]
f_el = 0.8 # Share of electricity demand which is converted to heat appliances



#################################
#### Function that loads data ###
#################################

def load_data_weather_buildings():
    
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py

    weather = pd.read_csv(os.path.join(path, "Weather.csv"),header=0,encoding = 'unicode_escape')
    weather.columns = ['Temp', 'Irr']

    buildings = pd.read_csv(os.path.join(path, "Buildings.csv"),header=0,encoding = 'unicode_escape')
    buildings.columns = ['Name', 'Year', 'Ground', 'Heat', 'Elec']

    return weather, buildings



#########################################################################################################
#### Function that returns profiles for people (in offices, canteens and classrooms) and electricity ####
#########################################################################################################

def occupancy_profile():

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
    
    # Yearly profile considering weekends for each usage (office, canteen and classroom)
    yearly_off=week_off*52 + occ_off  ##### Lundi *2
    yearly_class=week_class*52 + occ_class
    yearly_can=week_can*52 + occ_can
    yearly_elec=week_elec*52 + weekday_elec

    return [np.array(yearly_off),np.array(yearly_class),np.array(yearly_can),np.array(yearly_elec)]



##################################################################
#### Function that returnes the heat gain profile from people ####
##################################################################

def people_gains(office_profile,class_profile,cantine_profile):
    
    # Heat gains from people (Office, Restaurant, Classroom)
    heat_gain_off=5
    heat_gain_rest=35
    heat_gain_class=23.3
    heat_gain_others=0
    
    # Share areas (Office, Restaurant, Classroom)
    share_off=0.3
    share_rest=0.05
    share_class=0.35
    share_others=0.3
 
    q_people=heat_gain_off*share_off*office_profile + heat_gain_rest*share_rest*cantine_profile + heat_gain_class*share_class*class_profile
    return q_people #[W/m^2]



#################################################################
#### Function that returns the electricity heat gain profile ####
#################################################################

def elec_gains(building_id: str,elec_profile):
    q_elec=elec_profile*buildings[buildings['Name']==building_id]['Elec'].to_numpy()/3654*1000 #[W]
    return q_elec



######################################################
#### Main function that solves the Newton-Raphson ####
######################################################

def solving_NR(tolerance,max_iteration,building_id: str,k_sun_guess):
    
    # Initialize counters and tolerances
    error=1
    counter=0

    # Getting other parameters
    A_th=buildings[buildings['Name']==building_id]['Ground'].values[0] #[m^2]
    Q_year=buildings[buildings['Name']==building_id]['Heat'].values[0]*1000 #[Wh]
    temperature=weather.Temp.to_numpy()+273 #[K]
    irradiance=weather.Irr.to_numpy() #[W/m^2]
   
    # Mean values calculation (for second part of the function)


    ################# This is where we have an issue!!!!!##################

    ## Works, but wrong hypothesis
    irr_mean=irradiance.mean()
    q_people_mean=q_people.mean()
    q_elec_mean=q_elec.mean()

    ## Does not work but correct hypothesis (uncomment the 5 next lines if you want to try)

    #T_th_1=T_th-1
    #T_th_2=T_th+1
    #irr_mean=irradiance[(temperature>=T_th_1) & (temperature<=T_th_2) & (elec_profile==1)].mean()
    #q_people_mean=q_people[(temperature>=T_th_1) & (temperature<=T_th_2) & (elec_profile==1)].mean()
    #q_elec_mean=q_elec[(temperature>=T_th_1) & (temperature<=T_th_2) & (elec_profile==1)].mean()


    #########################################################################
    
    # Initialize guess values
    k_th=k_th_guess
    k_sun=(k_th*(T_int-T_th)-f_el*q_elec_mean/A_th-q_people_mean)/irr_mean
    k=np.array([k_th,k_sun])

 
    #### Main while loop for NR ####

    while ((np.abs(error)>=tolerance) and (counter<max_iteration)):

        #### Construction of the function ####

        # Determine Q(t), and sum all that respect the conditions
        Q_t=A_th*(k[0]*(T_int-temperature)-k[1]*irradiance-q_people)-f_el*q_elec
        Q_tot=Q_t[(temperature<=T_th) & (Q_t>=0) & (elec_profile==1)].sum()
        #Lastly, since the newton raphson model seeks solution for F(x)=0, we must substract the annual heating value to the first part of the function
        Q_tot=Q_tot-Q_year

        # Compute the derivative the same way
        deriv=A_th*(T_int-temperature)
        deriv_tot=deriv[(temperature<=T_th) & (Q_t>=0) & (elec_profile==1)].sum()


        #### Solving with the Newton Raphson method ####

        k_th_new=k[0]-Q_tot/deriv_tot
        k_sun_new=(k_th_new*(T_int-T_th)-f_el*q_elec_mean/A_th-q_people_mean)/irr_mean
        k=np.array([k_th_new,k_sun_new])

        
        #### Loop flags computation ####

        # Compute again the function with the new k_th and k_sun, to get the error
        Q_t=A_th*(k[0]*(T_int-temperature)-k[1]*irradiance-q_people)-f_el*q_elec
        Q_tot=Q_t[(temperature<=T_th) & (Q_t>=0) & (elec_profile==1)].sum()
        error=Q_tot-Q_year

        # Increment the loop counter
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

######################################################
#### Function for clustering ####
######################################################



#Function to compute the clustering error between the Qthbase and the Qth_cluster where Qthbase it the heat demand for the 8760 hours and Qth_cluster is the heat demand for the 8760 hours after clustering with n clusters

def clustering_error(Qth_base, Qth_cluster, heat, heatcluster):
    #compute Qth_base and Qth_cluster annualy for each buildings column(i.e sum of each column)
    Qth_base = heat.sum(axis=0)
    Qth_cluster = heatcluster.sum(axis=0)

    #Compute total annual Q demand for the entire EPFL
    Qtot_base = Qth_base.sum() 
    Qtot_cluster = Qth_cluster.sum()
    error = (Qtot_base - Qtot_cluster) / Qtot_base
    return error

###################################
#### Getting all work together ####
###################################

if __name__ == '__main__': 
    # the code below will be executed only if you run the NR_function.py file as main file, not if you import the functions from another file (another .py or .qmd)
    
    # Load data
    weather, buildings = load_data_weather_buildings()

    #Compute profile and heat gains from people
    [office_profile,class_profile,cantine_profile,elec_profile] = occupancy_profile()
    q_people = people_gains(office_profile,class_profile,cantine_profile)
    
    # State required tolerances and maximum number of iterations
    tolerance=0.00001
    max_iteration=100

    # State initial guesses for k_th
    k_th_guess=5
  
    # Initialize array to record values for each building
    k_th=np.zeros(len(buildings))
    k_sun=np.zeros(len(buildings))
    number_iteration=np.zeros(len(buildings))
    error1=np.zeros(len(buildings))
    error2=np.zeros(len(buildings))
    spec_elec=np.zeros(len(buildings))
    floor_area=np.zeros(len(buildings))
    specQ_people=np.ones(len(buildings))*q_people.mean()
    

    # Loop to get values for each building
    count=0
    Q_th=np.zeros([8760,len(buildings)])
    for building_id in buildings['Name']:
        q_elec=elec_gains(building_id, elec_profile)
        [[k_th[count],k_sun[count]],number_iteration[count],error1[count]]=solving_NR(tolerance,max_iteration,building_id,k_th_guess)
        spec_elec[count]=buildings[buildings['Name']==building_id]['Elec'].values[0]/3654/buildings[buildings['Name']==building_id]['Ground'].values[0]
        floor_area[count]=buildings[buildings['Name']==building_id]['Ground'].values[0]
        Q_temp=floor_area[count]*(k_th[count]*(T_int-(weather.Temp.to_numpy()+273))-k_sun[count]*weather.Irr.to_numpy()-q_people)-f_el*q_elec
        Q_temp[(Q_temp<=0) | ((weather.Temp.to_numpy()+273)>T_th) | (elec_profile!=1)]=0
        Q_th[:,count]=Q_temp/1000
        
        count=count+1

    #Construct dataframe for Q_th(t) for each building

    heat=pd.DataFrame(Q_th, columns=buildings['Name'].to_numpy())

  
    #Storing everything in a pandas dataframe
    data={'Name':buildings['Name'].to_numpy(), 'FloorArea':floor_area, 'specElec':spec_elec, 'k_th': k_th/1000, 'k_sun':k_sun,'specQ_people':specQ_people/1000} #Note that there are 2 errors. This is because the function is bidimmensional
    solution=pd.DataFrame(data)

    #Saving dataframes in thermal_properties.csv and heat.csv
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    solution.to_csv(os.path.join(path, "thermal_properties.csv"),index=False)
    heat.to_csv(os.path.join(path, "heat.csv"),index=False)

    #Printing solutions
    #print(Solution)
    #print(Heat)

    #Plot Q_th for some building
    #ax=plt.plot(Q_th[:,4],'.')
    #plt.yscale('log')
    #plt.show()

    ##### USE OF CLUSTERING POINTS TO GET QTH####


    #import cluster with Irr and Text values for the cluster centers
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    clusterdf = pd.read_csv(os.path.join(path, "clusters.csv"),header=0,encoding = 'unicode_escape')

    count = 0
    Qthcluster = np.zeros([len(clusterdf),len(buildings)])
    delta_hr=1 #time step in hours

    for building_id in buildings['Name']:
        q_elec=elec_gains(building_id, elec_profile)
        [[k_th[count],k_sun[count]],number_iteration[count],error1[count]]=solving_NR(tolerance,max_iteration,building_id,k_th_guess)
        spec_elec[count]=buildings[buildings['Name']==building_id]['Elec'].values[0]/3654/buildings[buildings['Name']==building_id]['Ground'].values[0]
        floor_area[count]=buildings[buildings['Name']==building_id]['Ground'].values[0]
        #use the clustering points to compute Qthcluster
        for i in range(len(clusterdf)):
            Q_temp=clusterdf.hours[i]*floor_area[count]*(k_th[count]*(T_int-(clusterdf.Temp[i]+273))-k_sun[count]*clusterdf.Irr[i]-q_people[i])-f_el*spec_elec[count]
            Qthcluster[i,count]=Q_temp/1000
        count=count+1

     

    #Construct dataframe for Qthcluster(t) for each building
    heatcluster=pd.DataFrame(Qthcluster, columns=buildings['Name'].to_numpy())

    #Concate clusterdf and heatcluster dataframes
    df_task1=pd.concat([clusterdf,heatcluster],axis=1)

    #Saving dataframes in final_df_task1.csv (NB: index is present in the csv file)
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    df_task1.to_csv(os.path.join(path, "final_df_task1.csv"),index=True)

    #Saving another dataframe with only the buldings with Constrcution_year = 1 (i.e.medium Temp heating demand)
    
    # Filter the rows where 'Year' is equal to 1
    filtered_buildings = buildings[buildings['Year'] == 1]

    # Create a list of building names to keep
    building_names_to_keep = filtered_buildings['Name'].tolist()

    # Filter the columns in 'heatcluster' based on the list of building names to keep
    df_task1_medium = heatcluster[building_names_to_keep]
    ##Concate the new medium dataframe with clusterdf 
    df_task1_medium = pd.concat([clusterdf,df_task1_medium],axis=1)
    
    #Saving dataframes in final_df_task1_reduced.csv (NB: index is present in the csv file)
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    df_task1_medium.to_csv(os.path.join(path, "final_df_task1_medium.csv"),index=True)



    #Computing the error between Qthbase and Qthcluster
    error = clustering_error(Q_th, Qthcluster, heat, heatcluster)
    print(error)

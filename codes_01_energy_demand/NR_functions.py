
import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
import os

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

def people_gains(profile_class, profile_rest, profile_off):
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

def solving_NR(tolerance,max_iteration,building_id, k_th_guess, k_sun_guess):
    
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

    #Compute mean values for irradiance, heat gain from people and appliances when around the cutoff temperature
    cutoff_indicator = ((T_th -1 <= T_ext) & (T_ext <= T_th + 1)) # state cutoff condition : T_ext is in [T_th-1, T_th+1] 

    q_elec_mean = q_elec[cutoff_indicator].mean() 
    q_people_mean = q_people[cutoff_indicator].mean()
    irr_mean = irr[cutoff_indicator].mean()

    specQ_people = q_people.mean()
    # Newton Raphson method
    
    alpha = 0.7 # relaxation factor

    while iteration < max_iteration:

        # Build f(k_th,k_sun) = 0
        h = (k_th*(T_int-T_ext)- q_people - k_sun*irr - q_elec*f_el) #auxiliary function to determine the heating mode

        heating_indicator = ((T_ext <= T_th) & (h>=0) & (profile_elec > 0 )) # state heating mode conditions

        f = h[heating_indicator].sum() - Q_th/A_th # filter out negative values of hx and sum the remaining values

        # Build g(k_th,k_sun) = 0

        g = (k_th*(T_int-T_th) - k_sun*irr_mean - q_people_mean - q_elec_mean*f_el)
   
        # Build Jacobian
        J = np.array([[(T_int - T_ext[heating_indicator]).sum(),  -irr[heating_indicator].sum()],
                      [(T_int - T_th),                            -irr_mean]])
        
        # Solve the linear system using matrix inversion
        delta_k_th, delta_k_sun =  np.linalg.solve(J, [f, g])*(1-alpha)

        # Save old values to compute the relative error
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
    return k_th, k_sun, iteration, e_th, e_sun, A_th, specQ_people, q_elec.mean() 

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

    #Compute gains and profile
    profile_off, profile_class, profile_rest, profile_elec = occupancy_profile()

    # Compute typical operating conditions  with clustering

    n_clusters = 6
    #weatherCluster, model = weather_clustering(n_clusters)

    # State required tolerances and maximum number of iterations
    tol=1e-6
    max_iteration=1000

    # State initial guesses for k_th and k_sun
    k_th_guess=5
    k_sun_guess=1

    # Initialize array to record values for each building
    solution  = pd.DataFrame(columns=['FloorArea', 'specElec', 'k_th', 'k_sun', 'specQ_people'])
    Q_th = pd.DataFrame(columns=buildings['Name']) 
    #Q_th_cluster = pd.DataFrame(columns=model.get_feature_names_out())

    T_ext = weather.Temp + 273 # K
    irr = weather.Irr # W/m2
    
    for building_id in buildings['Name']:
        q_people = people_gains(profile_class, profile_rest, profile_off)
        q_elec = elec_gains(building_id, profile_elec)
        [k_th, k_sun, number_iteration, error1,error2, A_th, specQ_people, q_elec_mean] = solving_NR(tol,max_iteration,building_id,k_th_guess,k_sun_guess)
        solution.loc[building_id] = pd.Series({'FloorArea': A_th, 'specElec': q_elec_mean, 'k_th': k_th, 'k_sun': k_sun,'specQ_people': specQ_people})
        # Recompute hourly energy demands
        Q_th[building_id] = A_th*(k_th*(T_int-T_ext) - q_people - k_sun*irr - q_elec*f_el)

    heating_indicator = (((Q_th >= 0).all(axis=1)) & (T_ext <= T_th) & (profile_elec > 0)) # filter heat demands only
    Q_th = Q_th[heating_indicator]/1000 # convert to kWh
    #Q_th_cluster = Q_th.groupby(weather['Cluster'].loc[weather['Cluster']< n_clusters]).sum() # sum heat demands for each cluster
    
    #Saving dataframe in thermal_properties.csv
    PATH = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    
    #Q_typical.sum(axis=1).to_csv(os.path.join(PATH, "Q_typical.csv"),index=False)
    
    #solution.to_csv(os.path.join(PATH, "thermal_properties.csv"),index=False)

    #Printing solutions
    print('Solution = \n', solution)
    print('Q_th = \n', Q_th)
    #print('Q_th_cluster = \n', Q_th_cluster)

"""
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
    k_th_guess= 5
    k_sun_guess= 1
  
    # Initialize array to record values for each building
    k_th=np.zeros(len(buildings))
    k_sun=np.zeros(len(buildings))
    number_iteration=np.zeros(len(buildings))
    error1=np.zeros(len(buildings))
    error2=np.zeros(len(buildings))
    spec_elec=np.zeros(len(buildings))
    floor_area=np.zeros(len(buildings))
    specQ_people=np.zeros(len(buildings))
    
    #import cluster with Irr and Text values for the cluster centers
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    clusterdf = pd.read_csv(os.path.join(path, "clusters.csv"),header=0,encoding = 'unicode_escape')
    cluster_disagg=pd.read_csv(os.path.join(path, "clusters_dissaggregated.csv"),header=0)

    # Loop to get values for each building
    count=0
    Q_th=np.zeros([8760,len(buildings)])
    Q_th_cluster=np.zeros([len(clusterdf),len(buildings)])
    for building_id in buildings['Name']:
        q_elec=elec_gains(building_id, elec_profile)
        [k_th[count],k_sun[count],number_iteration[count],error1[count], error2[count], floor_area[count], specQ_people[count], spec_elec[count]]=solving_NR(tolerance,max_iteration,building_id,k_th_guess, k_sun_guess)
        #floor_area[count]=buildings[buildings['Name']==building_id]['Ground'].values[0]
        #spec_elec[count]=np.mean(q_elec[q_elec!=0])/floor_area[count]
        Q_temp=floor_area[count]*(k_th[count]*(T_int-(weather.Temp.to_numpy()+273))-k_sun[count]*weather.Irr.to_numpy()-q_people)-f_el*q_elec
        Q_temp[(Q_temp<=0) | (weather.Temp.to_numpy()+273>T_th) | (elec_profile!=1)]=0
        Q_th[:,count]=Q_temp/1000
        for i in range(len(clusterdf)):
            Q_th_cluster[i,count]=np.sum(Q_temp[cluster_disagg[cluster_disagg['cluster']==i]['Unnamed: 0'].values])/1000
        count=count+1

    #Construct dataframe for Q_th(t) for each building

    heat=pd.DataFrame(Q_th, columns=buildings['Name'].to_numpy())
  
    #Storing everything in a pandas dataframe
    data={'Name':buildings['Name'].to_numpy(), 'FloorArea':floor_area, 'specElec':spec_elec/1000, 'k_th': k_th/1000, 'k_sun':k_sun,'specQ_people':specQ_people/1000} #Note that there are 2 errors. This is because the function is bidimmensional
    solution=pd.DataFrame(data)

    #Saving dataframes in thermal_properties.csv and heat.csv
    #path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    #solution.to_csv(os.path.join(path, "thermal_properties.csv"),index=False)
    #heat.to_csv(os.path.join(path, "heat.csv"),index=False)

    #Printing solutions
    print(solution)
    print(heat)

    #Plot Q_th for some building
    #ax=plt.plot(Q_th[:,4],'.')
    #plt.yscale('log')
    #plt.show()

    
    ##### USE OF CLUSTERING POINTS TO GET QTH####
        #Done above

#    count = 0
#    Qthcluster = np.zeros([len(clusterdf),len(buildings)])
#    delta_hr=1 #time step in hours
#    q_people_cluster=np.zeros(len(clusterdf))
#    q_elec_cluster=np.zeros(len(clusterdf))

#    for building_id in buildings['Name']:
#        q_elec=elec_gains(building_id, elec_profile)
#        [[k_th[count],k_sun[count]],number_iteration[count],error1[count]]=solving_NR(tolerance,max_iteration,building_id,k_th_guess)
#        #spec_elec[count]=buildings[buildings['Name']==building_id]['Elec'].values[0]/3654/buildings[buildings['Name']==building_id]['Ground'].values[0]
#        floor_area[count]=buildings[buildings['Name']==building_id]['Ground'].values[0]
#        #use the clustering points to compute Qthcluster
#        for i in range(len(clusterdf)):
#            q_people_cluster[i]=np.mean(q_people[cluster_disagg[cluster_disagg['cluster']==i]['Unnamed: 0'].values])
#            q_elec_cluster[i]=np.mean(q_elec[cluster_disagg[cluster_disagg['cluster']==i]['Unnamed: 0'].values])
#            Q_temp=clusterdf.hours[i]*floor_area[count]*(k_th[count]*(T_int-(clusterdf.Temp[i]+273))-k_sun[count]*clusterdf.Irr[i]-q_people_cluster[i])-f_el*q_elec_cluster[i]
#            if (Q_temp<=0) | (clusterdf.Temp[i]+273>T_th):
#                Q_temp=0
#            Qthcluster[i,count]=Q_temp/1000
#        count=count+1

    #Construct dataframe for Qthcluster(t) for each building
    heatcluster=pd.DataFrame(Q_th_cluster, columns=buildings['Name'].to_numpy())

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
    error = clustering_error(Q_th, Q_th_cluster, heat, heatcluster)
    #print(error)
"""

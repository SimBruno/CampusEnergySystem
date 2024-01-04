
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
    """
    Load weather and building data from CSV files.

    Returns:
    - weather (pd.DataFrame): DataFrame containing weather data with columns 'Temp' (Temperature) and 'Irr' (Irradiance).
    - buildings (pd.DataFrame): DataFrame containing building data with columns 'Name', 'Year', 'Ground' (Ground area),
      'Heat' (Heat capacity), and 'Elec' (Electrical capacity).
    """
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py

    weather = pd.read_csv(os.path.join(path, "Weather.csv"),header=0,encoding = 'unicode_escape')
    weather.columns = ['Temp', 'Irr']

    buildings = pd.read_csv(os.path.join(path, "Buildings.csv"),header=0,encoding = 'unicode_escape')
    buildings.columns = ['Name', 'Year', 'Ground', 'Heat', 'Elec']

    return weather, buildings

def occupancy_profile():
    """
    Generate occupancy profiles for different building types and electricity consumption on weekdays.

    Returns:
    - profile_off (ndarray): Office occupancy profile.
    - profile_class (ndarray): Classroom occupancy profile.
    - profile_rest (ndarray): Restaurant occupancy profile.
    - profile_elec (ndarray): Electricity consumption profile on weekdays.
    """
    # Define occupancy and electricity profiles
    occ_off = [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.4, 0.6, 0.8, 0.8, 0.4, 0.2, 0, 0, 0, 0, 0]
    occ_class = [0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.6, 1, 1, 0.8, 0.2, 0.6, 1, 0.8, 0.8, 0.4, 0, 0, 0, 0, 0]
    occ_can = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.2, 0.4, 1, 0.4, 0.2, 0.4, 0, 0, 0, 0, 0, 0, 0, 0]
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
    """
    Calculate heat gains from people in different building types based on their occupancy profiles.

    Parameters:
    - profile_class: occupancy profile for the classroom building type.
    - profile_rest: occupancy profile for the restaurant building type.
    - profile_off: occupancy profile for the office building type.

    Returns:
    - people_gains: calculated heat gains from people for each corresponding hour.

    The function calculates the heat gains from people by multiplying the occupancy profiles for each building type 
    with their respective scaling factors. The scaling factors are determined by the product of the heat gains from people 
    and the share areas of each building type. The final result is the sum of the products for all building types.

    Note:
    - The input arrays (profile_class, profile_rest, profile_off) should have the same length.
    - The function assumes that the input profiles are provided for each hour and follow the same time sequence.
    - The units of heat gains, share areas, and resulting people gains are not specified in the code.
    """
    # Heat gains from people [W/m^2] (Office, Restaurant, Classroom)
    hg_off=5
    hg_rest=35
    hg_class=23.3
    hg_others=0
    
    # Share areas [-] (Office, Restaurant, Classroom)
    A_off=0.3
    A_rest=0.05
    A_class=0.35
    A_others=0.3

    # Scaling factors [-]
    sf_off = hg_off*A_off
    sf_rest = hg_rest*A_rest
    sf_class = hg_class*A_class
    sf_others = hg_others*A_others

    # Heat gains [W/m^2]
    return sf_off*profile_off + sf_rest*profile_rest + sf_class*profile_class 

def elec_gains(building_id, buildings , profile_elec):
    #elec_build=buildings.Elec ###Wh
    cf = 1000/profile_elec.sum() # Conversion factor from Wh to W and capacity factor
    
    # Elec gains [W/m^2]
    return cf*profile_elec*buildings.loc[buildings.Name==building_id].apply(lambda x: x['Elec']/x['Ground'] , axis=1).to_numpy()[0] # W/m2 for each hour of the year

def solving_NR(building_id, buildings, weather, q_elec, q_people, profile_elec, tolerance=1e-6,max_iteration=1000, k_th_guess=5, k_sun_guess=1):
    
    # Initialize counters and tolerances
    e_th = 1
    e_sun = 1
    iteration=0

    # Initialize guess values
    k_th = k_th_guess # W/(m^2 K)
    k_sun = k_sun_guess # W/(m^2 K)

    # Getting other parameters
    A_th=buildings['Ground'].loc[buildings.Name==building_id].values[0] # [m^2]
    Q_th=buildings['Heat'].loc[buildings.Name==building_id].values[0]*1000 # [kWh] -->  [Wh]
    T_ext=weather.Temp + 273 # K
    irr=weather.Irr # W/m2

    #Compute mean values for irradiance, heat gain from people and appliances when around the cutoff temperature
    cutoff_indicator = ((T_th - 1 <= T_ext) & (T_ext <= T_th + 1)) # state cutoff condition : T_ext is in [T_th-1, T_th+1] 

    q_elec_mean = q_elec[cutoff_indicator].mean() 
    q_people_mean = q_people[cutoff_indicator].mean()
    irr_mean = irr[cutoff_indicator].mean()

    #specQ_people = q_people.mean()
    specQ_people = q_people.sum()/profile_elec.sum()
    specElec     = q_elec.sum()/profile_elec.sum()
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
    # k_th [W/(m^2 K)], k_sun [-], number of iterations, error1, error2, A_th [m^2], specQ_people [W/m^2], q_elec_mean [W/m^2], heating_indicator [bool]
    return k_th, k_sun, iteration, e_th, e_sun, A_th, specQ_people, specElec, heating_indicator 

######################################################
#### Function for clustering ####
######################################################

def preprocess_data(weather, profile_elec, T_th=273+16):
    typeA = ((profile_elec > 0) & (weather.Temp + 273 <= T_th))
    typeB = ((profile_elec == 0) | (weather.Temp + 273 > T_th))
    typeO = (weather.Temp ==weather.Temp.min())

    weather.loc[typeA, 'Type'] = 'A'
    weather.loc[typeB, 'Type'] = 'B'
    weather.loc[typeO, 'Type'] = 'O'

    return weather

class WeatherClustering:
    def __init__(self, weather, n_clusters, profile_elec):
        self.weather = weather
        self.n_clusters = n_clusters
        self.profile_elec = profile_elec

    def preprocess_data(self, T_th=273+16):
        typeA = ((self.profile_elec > 0) & (self.weather.Temp + 273 <= T_th))
        typeB = ((self.profile_elec == 0) | (self.weather.Temp + 273 > T_th))
        typeO = (self.weather.Temp ==self.weather.Temp.min())

        self.weather_norm = self.weather[['Temp','Irr']].apply(lambda x: (x - x.mean()) / x.std())
        self.weather_norm.loc[typeA, 'Type'] = 'A'
        self.weather_norm.loc[typeB, 'Type'] = 'B'
        self.weather_norm.loc[typeO, 'Type'] = 'O'

        self.weather.loc[typeA, 'Type'] = 'A'
        self.weather.loc[typeB, 'Type'] = 'B'
        self.weather.loc[typeO, 'Type'] = 'O'

        self.data = self.weather_norm[['Temp', 'Irr']].loc[self.weather_norm.Type == 'A']

    def clustering(self, method):
        
        # Fit the clustering algorithms
        self.cluster = method.fit(self.data)

        # Retrieve the cluster labels
        self.labels = self.cluster.labels_

        # Retrieve unormalized cluster centers
        self.get_cluster_center(unormalized=True)

        # Compute SSE
        self.compute_sse()

    def get_cluster_center(self,unormalized=True):
        self.cluster_centers = np.zeros((self.n_clusters, self.data.shape[1]))

        for cluster_label in range(self.n_clusters):
            cluster_points = self.data[self.labels == cluster_label]
            self.cluster_center = np.mean(cluster_points, axis=0)
            self.cluster_centers[cluster_label] = self.cluster_center
        if unormalized:
            self.cluster_centers = self.cluster_centers*self.weather[['Temp','Irr']].std().values + self.weather[['Temp','Irr']].mean().values

    def compute_sse(self):
        self.sse = 0
        for i in range(len(self.cluster_centers)):
            cluster_points = self.data[self.labels == i]
            if len(cluster_points) > 0:
                squared_distances = np.sum((cluster_points - self.cluster_centers[i]) ** 2, axis=1)
                self.sse += np.sum(squared_distances)

    def plot_clusters(self,axs):
        sns.scatterplot(data=self.weather.loc[self.weather.Type == 'A'], x='Temp', y='Irr', hue=self.labels, palette='deep',ax=axs)
        sns.scatterplot(x=self.cluster_centers[:, 0], y=self.cluster_centers[:, 1], marker='o', color='black', s=100,ax=axs)

def clusteringCorentin(weather, n_clusters):

    # Input : weather dataframe with Temp, Irr and Type columns
    #         number of clusters
    # Output : weather dataframe with Temp, Irr, Type and Cluster columns
    #          cluster dataframe with cluster centers and number of hours per cluster

    #normalize weather data for each column with gaussian normalization
    weatherNorm = weather[['Temp','Irr']].apply(lambda x: (x - x.mean()) / x.std())
    
    # select only type A data to be clustered
    weatherNorm= weatherNorm.loc[weather.Type == 'A']
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(weatherNorm)

    #add cluster labels to the data
    weatherNorm['Cluster'] = kmeans.labels_
    weather['Cluster'] = weatherNorm['Cluster']
    weather['Cluster'].loc[weather.Type == 'O'] = n_clusters # add cluster label for outlier
    
    # retrieve cluster centers and unormalize them
    cluster = pd.DataFrame(kmeans.cluster_centers_,columns=['Temp','Irr'])
    cluster[['Temp','Irr']] = cluster[['Temp','Irr']]*weather[['Temp','Irr']].std() + weather[['Temp','Irr']].mean()
    # add extreme cluster center
    cluster.loc[n_clusters] = weather[['Temp','Irr']].min()
    # get operating hours per cluster
    cluster['Hours'] = weather['Cluster'].value_counts()
    
    return weather, cluster

def elbow_method(data, max_clusters):
    # Elbow method
    inertias = []
    for k in range(1, max_clusters+1):
        cluster = KMeans(n_clusters=k, random_state=0, n_init = 'auto').fit(data)
        inertias.append(cluster.inertia_)
    plt.plot(range(1, max_clusters+1), inertias, marker='o')
    plt.title('Elbow method for Kmeans')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

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
    PATH = os.path.dirname(__file__) # the path to codes_01_energy_demand.py

    # Compute k_th and k_sun

    # Load data
    weather, buildings = load_data_weather_buildings()

    # Compute gains and profile
    profile_off, profile_class, profile_rest, profile_elec = occupancy_profile()

    # State required tolerances and maximum number of iterations
    tol           = 1e-6
    max_iteration = 1000

    # State initial guesses for k_th and k_sun
    k_th_guess    = 5
    k_sun_guess   = 1

    # Initialize array to record values for each building
    thermal_properties  = pd.DataFrame(columns=['FloorArea', 'specElec', 'k_th', 'k_sun', 'specQ_people'])
    Q_th                = pd.DataFrame(columns=buildings['Name'])

    T_ext        = weather.Temp + 273 # K
    irr          = weather.Irr # W/m2

    #Precompute q_people since it's constant across all buildings
    q_people = people_gains(profile_class, profile_rest, profile_off)

    for building_id in buildings['Name']:
        #Estimate k_th and k_sun for each building
        q_elec = elec_gains(building_id, buildings, profile_elec)
        [k_th, k_sun, number_iteration, error1, error2, A_th, specQ_people, specElec, heating_indicator] = solving_NR(building_id, buildings, weather, q_elec, q_people, profile_elec)
        thermal_properties.loc[building_id] = pd.Series({'FloorArea': A_th, 'specElec': specElec/1000, 'k_th': k_th/1000, 'k_sun': k_sun,'specQ_people': specQ_people/1000})

        #Compute Q_th for each hour
        Q_th[building_id] = A_th*(k_th*(T_int-T_ext) - q_people - k_sun*irr - q_elec*f_el)/1000 # Wh--> kWh
        Q_th[heating_indicator==0] = 0 # Set Q_th to be only heat demand


    print('Thermal properties of the buildings: \n', thermal_properties)

    print('Total Q_th recomputed = ', Q_th.sum().sum(),'.\n')

    # Compute typical operating conditions  with clustering
    n_clusters         = 6
    weather = preprocess_data(weather, profile_elec)
    weather, cluster = clusteringCorentin(weather,n_clusters=6)

    Q_th_cluster        = pd.DataFrame(columns=buildings.Name)
    T_cluster    = cluster['Temp'] + 273              # C° --> K
    irr_cluster  = cluster['Irr']/1000                # W/m^2 --> kW/m^2 
  
    # Extract thermal_properties of buildings
    A_th            = thermal_properties['FloorArea']    # m^2
    k_th            = thermal_properties['k_th']         # kW/m^2 K
    k_sun           = thermal_properties['k_sun']        # -
    specQ_people    = thermal_properties['specQ_people'] # kW/m^2
    specElec        = thermal_properties['specElec']     # kW/m^2


    # Recompute thermal load for each cluster using cluster centers
    for building_id in buildings['Name']:
        Q_th_cluster[building_id] = A_th[building_id]*(k_th[building_id]*(T_int-T_cluster) - k_sun[building_id]*irr_cluster - specQ_people[building_id] - specElec[building_id]*f_el) # [kWh]

    Q_th_cluster[Q_th_cluster < 0]  = 0                                                         # Set negative values to 0
    Q_th_cluster                    = Q_th_cluster[buildings['Name'].loc[buildings.Year == 1]]  # Select only medium temp buildings
    Q_th_cluster                    = Q_th_cluster.sum(axis=1)                                  # Get total hourly demand per cluster
    Q_th_cluster                    = Q_th_cluster*cluster['Hours']                             # Get annual demand

    cluster['Q_th'] = Q_th_cluster        
    
    print('Cluster values: \n', cluster)

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

def elec_gains(building_id, buildings , profile_elec):
    #elec_build=buildings.Elec ###Wh
    cf = 1000/profile_elec.sum() # Conversion factor from Wh to W and capacity factor
    return cf*profile_elec*buildings.loc[buildings.Name==building_id].apply(lambda x: x['Elec']/x['Ground'] , axis=1).to_numpy()[0] # W/m2 for each hour of the year

def solving_NR(building_id, buildings, weather, q_elec, q_people, profile_elec, tolerance=1e-6,max_iteration=1000, k_th_guess=5, k_sun_guess=1):
    
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
    cutoff_indicator = ((T_th - 1 <= T_ext) & (T_ext <= T_th + 1)) # state cutoff condition : T_ext is in [T_th-1, T_th+1] 

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
    return k_th, k_sun, iteration, e_th, e_sun, A_th, specQ_people, q_elec.mean(), heating_indicator 

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
    solution  = pd.DataFrame(columns=['FloorArea m^2', 'specElec kWh/m^2', 'k_th kWh/m^2/K', 'k_sun', 'specQ_people kWh/m^2'])
    Q_th = pd.DataFrame(columns=buildings['Name']) 
    #Q_th_cluster = pd.DataFrame(columns=model.get_feature_names_out())

    T_ext = weather.Temp + 273 # K
    irr = weather.Irr # W/m2

    PATH = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
    cluster_df=pd.read_csv(os.path.join(PATH,'clusters_dissaggregated.csv')).rename(columns={'Unnamed: 0':'Time'})

    Q_extreme=[]
    for building_id in buildings['Name']:
        q_people = people_gains(profile_class, profile_rest, profile_off)
        q_elec = elec_gains(building_id, buildings, profile_elec)
        [k_th, k_sun, number_iteration, error1,error2, A_th, specQ_people, q_elec_mean, heating_indic] = solving_NR(building_id, buildings, weather, q_elec, q_people, profile_elec)
        solution.loc[building_id] = pd.Series({'FloorArea m^2': A_th, 'specElec kWh/m^2': q_elec_mean/1000, 'k_th kWh/m^2/K': k_th/1000, 'k_sun': k_sun,'specQ_people kWh/m^2': specQ_people/1000})
        # Recompute hourly energy demands
        Q_temp= A_th*(k_th*(T_int-T_ext) - q_people - k_sun*irr - q_elec*f_el)/1000
        Q_extreme.append(A_th*(k_th*(T_int-(273-9.2)) - specQ_people - q_elec_mean*f_el)/1000)
        Q_temp[heating_indic==False]=0
        Q_th[building_id]=Q_temp
        #for i in cluster_df['cluster'].unique():
    
    # Concatenate into clusters
    Q_th_cluster=pd.concat([pd.DataFrame(data=Q_th.iloc[cluster_df['Time']].assign(cluster=cluster_df['cluster'].astype(int)).groupby('cluster').apply(lambda x: x.sum(axis=0)).values, columns=buildings['Name'].tolist()+['cluster']).drop('cluster',axis=1),pd.DataFrame(data=np.reshape(np.array(Q_extreme),(1,len(buildings))),columns=buildings['Name'].tolist())],ignore_index=True)
   

    # Save The DFs in csv
    Q_th_cluster.drop(columns=buildings.query('Year==2')['Name'].values).to_csv(os.path.join(PATH, "Q_cluster_medium.csv"),index=True)
    solution.to_csv(os.path.join(PATH, "thermal_properties_final.csv"),index=False)

    



    
    #heating_indicator = (((Q_th >= 0).all(axis=1)) & (T_ext <= T_th) & (profile_elec > 0)) # filter heat demands only
    #Q_th = Q_th[heating_indicator]/1000 # convert to kWh
    #Q_th_cluster = Q_th.groupby(weather['Cluster'].loc[weather['Cluster']< n_clusters]).sum() # sum heat demands for each cluster
    
    #Saving dataframe in thermal_properties.csv
    
    
    #Q_typical.sum(axis=1).to_csv(os.path.join(PATH, "Q_typical.csv"),index=False)
    
    #solution.to_csv(os.path.join(PATH, "thermal_properties.csv"),index=False)

    #Printing solutions
    print('Solution = \n', solution)
    
    print('Q_th = \n', Q_th)
    

    
    


    #print('Q_th_cluster = \n', Q_th_cluster)
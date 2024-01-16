# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import matplotlib
matplotlib.use('Agg')
import codes_01_energy_demand.NR_functions as fct1
from codes_01_energy_demand.NR_functions import WeatherClustering
import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import os
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML
#
#
#
weather, buildings = fct1.load_data_weather_buildings()
h = 8760  # Hours in a year
T_th = 273 + 16 # Cut off temperature [K]
cp_air = 1152 # Air specific heat capacity [J/(m3.K)] 
T_int = 273 + 21 # Set point temperature [K]
air_new = 2.5 # Air renewal [m3/(m2.h)]
Vent = 0 # [...]
f_el = 0.8 # Share of electricity demand which is converted to heat appliances
#
#
#
#
#
#
#
#| label: df preview
buildings.head(2)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: generate profiles
profile_off, profile_class, profile_rest, profile_elec = fct1.occupancy_profile()
#
#
#
#
fig, axs = plt.subplots(1, 4, figsize=(14, 3))

profiles = [profile_off, profile_class, profile_rest, profile_elec]
titles = ['Office occupancy', 'Class occupancy', 'Restaurant occupancy', 'Electricity profile']
for i, profile in enumerate(profiles):
    #axs[i].plot(profile[72:96])
    axs[i].bar(range(0, 24), profile[72:96])
    axs[i].set_xticks(np.arange(1, 24, 2))
    axs[i].set_xticklabels(np.arange(1, 24, 2))
    axs[i].set_title(titles[i])
fig.tight_layout()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#Run the NR algorithm to produce the k_th and k_sun for every building
thermal_properties  = pd.DataFrame(columns=['FloorArea', 'specElec', 'k_th', 'k_sun', 'specQ_people'])
NR_info             = pd.DataFrame(columns=['error_k_th','error_k_sun','number_iteration'])
Q_th = pd.DataFrame(columns=buildings['Name']) 
T_ext = weather.Temp + 273 # K
irr = weather.Irr # W/m2

q_people = fct1.people_gains(profile_class, profile_rest, profile_off)

for building_id in buildings['Name']:
    #Estimate k_th and k_sun for each building
    q_elec = fct1.elec_gains(building_id, buildings, profile_elec)
    [k_th, k_sun, number_iteration, error1, error2, A_th, specQ_people, specElec, heating_indicator] = fct1.solving_NR(building_id, buildings, weather, q_elec, q_people, profile_elec)
    thermal_properties.loc[building_id] = pd.Series({'FloorArea': A_th, 'specElec': specElec/1000, 'k_th': k_th/1000, 'k_sun': k_sun,'specQ_people': specQ_people/1000})
    NR_info.loc[building_id] = pd.Series({'error_k_th': error1,'error_k_sun':error2,'number_iteration':number_iteration})


    #Compute Q_th for each hour
    Q_th[building_id] = A_th*(k_th*(T_int-T_ext) - q_people - k_sun*irr - q_elec*f_el)/1000 # Wh--> kWh
    Q_th[heating_indicator==0] = 0 # Set Q_th to be only heat demand

print(NR_info)

PATH = os.path.dirname(os.getcwd()) # the path to codes_01_energy_demand.py
thermal_properties.to_csv(os.path.join(PATH,"report-group-3","codes_01_energy_demand", "thermal_properties.csv"),index=False)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: weather cluster plotting

#Add 'Type' column
weather = fct1.preprocess_data(weather, profile_elec)

# Create the initial figure
fig_weather = px.scatter(weather, x=weather.index, y='Temp', color='Type', title='Weather data - Temperature (processed)',
                         labels={'Temp': 'Temperature [°C]', 'Hours': 'Hour'})

# Use a context manager to delay the display
with fig_weather.batch_update():

    fig_weather.update_xaxes(title='Hour')

    # Add horizontal line at 16°C
    fig_weather.add_shape(type='line', x0=0, x1=len(weather), y0=16, y1=16, line=dict(color='red'))

    # Write "16°C" next to the line
    fig_weather.add_annotation(text='16°C', xref='paper', yref='y', x=0, y=17, showarrow=False, font=dict(color='red'))

    # Show the legend
    fig_weather.update_layout(legend_title_text='Type')

fig_weather = px.scatter(weather, x=weather.index, y='Irr', color='Type', title='Weather data - Irradiation (processed)',
                         labels={'Irr': 'Irradiation [W/m2]', 'Hours': 'Hour'})
                         
fig_weather.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: weather cluster plotting

#Add 'Type' column
weather = fct1.preprocess_data(weather, profile_elec)

# Create the initial figure
fig_weather = px.scatter(weather, x=weather.index, y='Temp', color='Type', title='Weather data - Temperature (processed)',
                         labels={'Temp': 'Temperature [°C]', 'Hours': 'Hour'})

# Use a context manager to delay the display
with fig_weather.batch_update():

    fig_weather.update_xaxes(title='Hour')

    # Add horizontal line at 16°C
    fig_weather.add_shape(type='line', x0=0, x1=len(weather), y0=16, y1=16, line=dict(color='red'))

    # Write "16°C" next to the line
    fig_weather.add_annotation(text='16°C', xref='paper', yref='y', x=0, y=17, showarrow=False, font=dict(color='red'))

    # Show the legend
    fig_weather.update_layout(legend_title_text='Type')

fig_weather = px.scatter(weather, x=weather.index, y='Irr', color='Type', title='Weather data - Irradiation (processed)',
                         labels={'Irr': 'Irradiation [W/m2]', 'Hours': 'Hour'})
                         
fig_weather.show()
#
#
#
#
#
#
#
#
#
#| label: fig-clusteringComp
#| fig-cap: Different clustering method results

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
spectral = SpectralClustering(assign_labels='discretize', n_clusters=n_clusters, random_state=0)
agglo = AgglomerativeClustering(n_clusters=n_clusters)

weather_clustering = WeatherClustering(weather, n_clusters, profile_elec)
weather_clustering.preprocess_data()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

titles = ['Spectral Clustering','Agglomerative Clustering','KMeans Clustering']

fig.suptitle(f'Clustering comparison, {n_clusters} clusters')

for i, method in enumerate([spectral, agglo, kmeans]):
    weather_clustering.clustering(method)
    weather_clustering.plot_clusters(axs[i])
    axs[i].set_title(titles[i])

plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: fig-elbowMethod
#| 
fct1.elbow_method(weather_clustering.data,15)
#
#
#
#
#
#
#
#
#| label: perform clustering

weather, cluster = fct1.clusteringCorentin(weather,n_clusters=6)

Q_th_cluster = pd.DataFrame(columns=buildings.Name)
# Extract thermal_properties of buildings
A_th            = thermal_properties['FloorArea']    # m^2
k_th            = thermal_properties['k_th']         # kW/m^2 K
k_sun           = thermal_properties['k_sun']        # -
specQ_people    = thermal_properties['specQ_people'] # kW/m^2
specElec        = thermal_properties['specElec']     # kW/m^2

T_cluster       = cluster['Temp'] + 273              # C° --> K
irr_cluster     = cluster['Irr']/1000                # W/m^2 --> kW/m^2 

# Recompute thermal load for each cluster using cluster centers
for building_id in buildings['Name']:
    Q_th_cluster[building_id] = A_th[building_id]*(k_th[building_id]*(T_int-T_cluster) - k_sun[building_id]*irr_cluster - specQ_people[building_id] - specElec[building_id]*f_el) # [kWh]

Q_th_cluster[Q_th_cluster < 0]  = 0                                                         # Set negative values to 0
Q_th_cluster                    = Q_th_cluster[buildings['Name'].loc[buildings.Year == 1]]  # Select only medium temp buildings
Q_th_cluster                    = Q_th_cluster.sum(axis=1)                                  # Get total hourly demand per cluster
Q_th_cluster                    = Q_th_cluster*cluster['Hours']                             # Get annual demand

cluster['Q_th'] = Q_th_cluster

cluster.to_csv(os.path.join(PATH,"report-group-3","codes_01_energy_demand", "clusters_data.csv"), index=False)

#
#
#
#| label: plot clustering 
# Scatter plot for weather data type A
fig = px.scatter(weather.loc[weather['Type']=='A'], x='Temp', y='Irr', color="Cluster", 
                 title='Weather data type A', labels={'Temp': 'Temperature [°C]', 'Irr': 'Irradiation [W/m2]'})
cluster_trace = go.Scatter(
    x=cluster.iloc[:, 0],
    y=cluster.iloc[:, 1],
    mode='markers',
    marker=dict(color='red', size=20, opacity=0.8),
    name='Cluster Centroids'
)
fig.add_trace(cluster_trace)

cluster.rename(columns={
    'Temp': 'Temperature [°C]',
    'Irr': 'Irradiance [W/m²]',
    'Hours': 'Operating time [h]',
    'Q_th' : 'Heat load [kWh/year]'
}, inplace=True)
html_table = cluster.round(2).to_html()
title_html = '<h4 style="text-align:center;">Cluster Centroids</h4>'
final_html = title_html + html_table
HTML(final_html)
#
#
#
#| label: compute clustering error
clustering_error = (buildings['Heat'].loc[buildings.Year == 1].sum()-Q_th_cluster.sum())/buildings['Heat'].loc[buildings.Year == 1].sum()*100
#
#
#
#
#

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
import codes_01_energy_demand.NR_functions as fct1
import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import os

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
profile_off, profile_class, profile_rest, profile_elec = fct1.occupancy_profile()
#
#
#
#
fig, axs = plt.subplots(1, 4, figsize=(14, 3))
axs[0].plot(profile_off[72:96])
axs[0].set_title('Office occupancy')
axs[1].plot(profile_class[72:96])
axs[1].set_title('Class occupancy')
axs[2].plot(profile_rest[72:96])
axs[2].set_title('Restaurant occupancy')
axs[3].plot(profile_elec[72:96])
axs[3].set_title('Electricity profile')
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
solution  = pd.DataFrame(columns=['FloorArea', 'specElec', 'k_th', 'k_sun', 'specQ_people'])
Q_th = pd.DataFrame(columns=buildings['Name']) 
T_ext = weather.Temp + 273 # K
irr = weather.Irr # W/m2
for building_id in buildings['Name']:
        q_people = fct1.people_gains(profile_class, profile_rest, profile_off)
        q_elec = fct1.elec_gains(building_id, buildings, profile_elec)
        [k_th, k_sun, number_iteration, error1,error2, A_th, specQ_people, q_elec_mean] = fct1.solving_NR(building_id, buildings, weather,q_elec,q_people,profile_elec)
        solution.loc[building_id] = pd.Series({'FloorArea': A_th, 'specElec': q_elec_mean, 'k_th': k_th, 'k_sun': k_sun,'specQ_people': specQ_people})
        # Recompute hourly energy demands
        Q_th[building_id] = A_th*(k_th*(T_int-T_ext) - q_people - k_sun*irr - q_elec*f_el)
print(solution.head(3))
#
#
#
#
#
#
#
#

n_clusters = 5

algos = {
    'kmeans': KMeans(n_clusters=n_clusters, random_state=0, n_init = 'auto'),
    'spectral': SpectralClustering(assign_labels='discretize', n_clusters=n_clusters, random_state=0),
    'agglo': AgglomerativeClustering(n_clusters=n_clusters)
}

for algo_name, algo_instance in algos.items():
    algo_name = fct1.WeatherClustering(weather, n_clusters, profile_elec)
    algo_name.preprocess_data()
    algo_name.clustering(algo_instance)

print(kmeans.sse)
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
import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML

#path = os.path.dirname("report-group-3\codes_01_energy_demand\_") # the path to codes_01_energy_demand.py
weather = pd.read_csv("codes_01_energy_demand/Weather.csv",header=0,encoding = 'unicode_escape')
weather.columns = ['Temp', 'Irr']
buildings = pd.read_csv("codes_01_energy_demand/Buildings.csv",header=0,encoding = 'unicode_escape')
buildings.columns = ['Name', 'Year', 'Ground', 'Heat', 'Elec']

# Plot temperature
fig_temp = px.scatter(x=weather.index, y=weather['Temp'], labels={'y': 'Temperature [°C]', 'x': 'Hour'},
                      title='Weather data - Temperature')
#fig_temp.show()

# Plot irradiation
fig_irr = px.scatter(x=weather.index, y=weather['Irr'], labels={'y': 'Irradiation [W/m2]', 'x': 'Hour'},
                     title='Weather data - Irradiation')
#fig_irr.show()

#cluster with sklearn using kmeans algorithm using 2 types: Type A) timesteps were the buildings are used (Monday to Friday 7am to 9pm) and the external temperature is below the cut off temperature of 16 °C.• Type B) the exact opposite of type A leaving the timesteps when the buildings are not used (night,weekend) and the external temperature is greater than 16 °C
#intialize the weather_df with columns names index, temperature, irradiation and type
weather_df = []

for k in range(0, len(weather.Temp)-1):
    if (k % 24 >= 7 and k % 24 <= 21) and weather.Temp[k] < 16 and (k % 168 <= 120):
            weather_df.append([k, weather.Temp[k], weather.Irr[k], 'A'])
    else:
        weather_df.append([k, weather.Temp[k], weather.Irr[k], 'B'])

weather_df = pd.DataFrame(weather_df, columns=['Hours', 'Temp', 'Irr', 'Type'])

# Add outliers type
mean_w = weather_df.Temp.mean()
std_w = weather_df.Temp.std()
# Z-score = 3
#ZSCORE = 2.5
#weather_df.loc[(weather_df.Temp < mean_w - ZSCORE*std_w) | (weather_df.Temp > mean_w + ZSCORE*std_w), 'Type'] = 'O'
#set the minimum temperature value of weather_df to state O. There is only one value
weather_df.loc[weather_df.Temp == weather_df.Temp.min(), 'Type'] = 'O'

#save weather_df in a new csv file
weather_df.to_csv('weather_df.csv', index=False)

# Assuming weather_df is your DataFrame

# Create the initial figure
fig_weather = px.scatter(weather_df, x='Hours', y='Temp', color='Type', title='Weather data - Temperature (processed)',
                         labels={'Temp': 'Temperature [°C]', 'Hours': 'Hour'})

# Use a context manager to delay the display
with fig_weather.batch_update():

    fig_weather.update_xaxes(title='Hour')

    # Add horizontal line at 16°C
    fig_weather.add_shape(type='line', x0=0, x1=len(weather_df), y0=16, y1=16, line=dict(color='red'))

    # Write "16°C" next to the line
    fig_weather.add_annotation(text='16°C', xref='paper', yref='y', x=0, y=17, showarrow=False, font=dict(color='red'))

    # Show the legend
    fig_weather.update_layout(legend_title_text='Type')

fig_weather = px.scatter(weather_df, x='Hours', y='Irr', color='Type', title='Weather data - Irradiation (processed)',
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
#
#
#
#
#
#
#
#
#

#select the data for type A
weather_A = weather_df[weather_df.Type == 'A']
weather_A = weather_A[['Temp', 'Irr']]
#select the data for type B
weather_B = weather_df[weather_df.Type == 'B']
weather_B = weather_B[['Temp', 'Irr']]
#select the data for outliers
weather_O = weather_df[weather_df.Type == 'O']
weather_O = weather_O[['Temp', 'Irr']]

weather_A_norm = []
#normalize weather_A data for each column with gaussian normalization
for i in range(0, len(weather_A.columns)):
    weather_A_norm.append((weather_A.iloc[:, i] - weather_A.iloc[:, i].mean())/weather_A.iloc[:, i].std())

weather_A_norm = pd.DataFrame(weather_A_norm).transpose()
#rename the columns
weather_A_norm.columns = ['Temp', 'Irr']

#cluster the data in 5 clusters
kmeans_A = KMeans(n_clusters=6, random_state=0, n_init=10).fit(weather_A_norm)
#add the cluster column to the dataframe
weather_A_norm['cluster'] = kmeans_A.labels_
weather_A['cluster'] = kmeans_A.labels_
#count the number of hours in each cluster
hours_per_cluster = weather_A.groupby('cluster').count().iloc[:, 0]
#create a cluster disaggregated dataframe (i.e. all data are not grouped by cluster)
cluster_disagg = weather_A

#load the data in a csv file
#path = os.path.dirname(__file__)
#cluster_disagg.to_csv(os.path.join(path,'clusters_dissaggregated.csv'),index=True)

cluster = kmeans_A.cluster_centers_
for c in cluster:
    c[0] = c[0]*weather_A.Temp.std() + weather_A.Temp.mean()
    c[1] = c[1]*weather_A.Irr.std() + weather_A.Irr.mean()

centroid_extreme = [weather_O.values[0][0], weather_O.values[0][1], 1]
#add centroid_extreme to the cluster

# Convert the NumPy array to a Pandas DataFrame
cluster_df = pd.DataFrame(cluster, columns=['Temp', 'Irr'])
# Add the number of hours per cluster
cluster_df['hours'] = hours_per_cluster.values

# Add the extreme centroid in cluster_df using pd.concat()
cluster_df = pd.concat([cluster_df, pd.DataFrame([centroid_extreme], columns=['Temp', 'Irr', 'hours'])], ignore_index=True)

#create a cluster.csv file with the centroids of the clusters and the numbers of hours in each cluster
#path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
#cluster_df.to_csv(os.path.join(path, "clusters.csv"),index=False)

# Scatter plot for weather data type A
fig = px.scatter(weather_A, x='Temp', y='Irr', color="cluster", 
                 title='Weather data type A', labels={'Temp': 'Temperature [°C]', 'Irr': 'Irradiation [W/m2]'})
cluster_trace = go.Scatter(
    x=cluster[:, 0],
    y=cluster[:, 1],
    mode='markers',
    marker=dict(color='red', size=20, opacity=0.8),
    name='Cluster Centroids'
)
fig.add_trace(cluster_trace)

#cluster_df = cluster_df.set_index('Point')
cluster_df.rename(columns={
    'Temp': 'Temperature [°C]',
    'Irr': 'Irradiance [W/m2]',
    'hours': 'Operating time [h]'
}, inplace=True)
cluster_df = cluster_df.round(2)
html_table = cluster_df.to_html()
title_html = '<h2 style="text-align:center;">Cluster Centroids</h2>'
final_html = title_html + html_table
HTML(final_html)
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
import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    header=dict(values=list(buildings.columns),
                fill_color='lightskyblue',
                line_color='darkslategray',
                align='center'),
    cells=dict(values=[buildings[col] for col in buildings],
               fill_color='white',
               line_color='darkslategray',
               align='center'))
])
fig.update_layout(width=800, height=700)

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
# Generate the yearly occupancy profile of the buildings
occ_profile = fct1.occupancy_profile()
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

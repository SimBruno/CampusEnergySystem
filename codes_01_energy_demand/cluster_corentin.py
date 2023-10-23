import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

def load_data_weather_buildings():
    
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py

    weather = pd.read_csv(os.path.join(path, "Weather.csv"),header=0,encoding = 'unicode_escape')
    weather.columns = ['Temp', 'Irr']



    buildings = pd.read_csv(os.path.join(path, "Buildings.csv"),header=0,encoding = 'unicode_escape')
    buildings.columns = ['Name', 'Year', 'Ground', 'Heat', 'Elec']

    return weather, buildings

weather, buildings = load_data_weather_buildings()


#plot weather in 2 figures one for temperature and one for irradiation
plt.figure()
plt.plot(weather.Temp, '+')
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data')

plt.figure()
plt.plot(weather.Irr, '+')
plt.xlabel('Hour')
plt.ylabel('Irradiation [W/m2]')
plt.title('Weather data')
#plt.show()


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
ZSCORE = 2.5
weather_df.loc[(weather_df.Temp < mean_w - ZSCORE*std_w) | (weather_df.Temp > mean_w + ZSCORE*std_w), 'Type'] = 'O'

#save weather_df in a new csv file
weather_df.to_csv('weather_df.csv', index=False)

#plot weather in 2 figures one for temperature and one for irradiation and 2 colours depending on type A or type B
plt.figure()
plt.plot(weather_df[weather_df.Type == 'A'].Temp, '+')
plt.plot(weather_df[weather_df.Type == 'B'].Temp, '+')
plt.plot(weather_df[weather_df.Type == 'O'].Temp, '+')
#plt.plot(weather_df.Temp)
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data')
#grid on
plt.grid(True)
#add horizontal line at 16°C
plt.axhline(y=16, color='r', linestyle='-')
#write "16°C" next to the line
plt.text(0, 17, '16°C', color='r')
#legend for the 3 types
plt.legend(['Type A', 'Type B', 'Outliers'], loc='upper right')


plt.figure()
plt.plot(weather_df[weather_df.Type == 'A'].Irr, '+')
plt.plot(weather_df[weather_df.Type == 'B'].Irr, '+')
plt.plot(weather_df[weather_df.Type == 'O'].Irr, '+')
#plt.plot(weather_df.Temp)
plt.xlabel('Hour')
plt.ylabel('Irradiation [W/m2]')
plt.title('Weather data')
#grid on
plt.grid(True)
#legend for the 3 types
plt.legend(['Type A', 'Type B', 'Outliers'], loc='upper right')
plt.show()

#cluster type A in 4 clusters
#select the data for type A
weather_A = weather_df[weather_df.Type == 'A']
#select the data for type B
weather_B = weather_df[weather_df.Type == 'B']
#select the data for outliers
weather_O = weather_df[weather_df.Type == 'O']

#select the data for type A
weather_A = weather_A[['Temp', 'Irr']]
#select the data for type B
weather_B = weather_B[['Temp', 'Irr']]
#select the data for outliers
weather_O = weather_O[['Temp', 'Irr']]

weather_A_norm = []
#normalize weather_A data for each column with gaussian normalization
for i in range(0, len(weather_A.columns)):
    weather_A_norm.append((weather_A.iloc[:, i] - weather_A.iloc[:, i].mean())/weather_A.iloc[:, i].std())

weather_A_norm = pd.DataFrame(weather_A_norm).transpose()
#rename the columns
weather_A_norm.columns = ['Temp', 'Irr']

#cluster the data in 5 clusters
kmeans_A = KMeans(n_clusters=6, random_state=0).fit(weather_A_norm)
#add the cluster column to the dataframe
weather_A_norm['cluster'] = kmeans_A.labels_
weather_A['cluster'] = kmeans_A.labels_
#count the number of hours in each cluster
hours_per_cluster = weather_A.groupby('cluster').count().iloc[:, 0]

#plot the data
plt.figure()
plt.scatter(weather_A.Temp, weather_A.Irr, c=weather_A.cluster)

cluster = kmeans_A.cluster_centers_
for c in cluster:
    c[0] = c[0]*weather_A.Temp.std() + weather_A.Temp.mean()
    c[1] = c[1]*weather_A.Irr.std() + weather_A.Irr.mean()



print(cluster)

# Convert the NumPy array to a Pandas DataFrame
cluster_df = pd.DataFrame(cluster, columns=['Temp', 'Irr'])
# Add the number of hours per cluster
cluster_df['hours'] = hours_per_cluster.values


#create a cluster.csv file with the centroids of the clusters and the numbers of hours in each cluster
path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py
cluster_df.to_csv(os.path.join(path, "clusters.csv"),index=False)

#plot the centroids in red
plt.scatter(cluster[:, 0], cluster[:, 1], c='red', s=200, alpha=0.5)
plt.xlabel('Temperature [°C]')
plt.ylabel('Irradiation [W/m2]')
plt.title('Weather data type A')

'''
#print the centroids of the clusters
print(kmeans_A.cluster_centers_)
#print the labels of the clusters
print(kmeans_A.labels_)
#print the inertia of the clusters
print(kmeans_A.inertia_)
#print the number of iterations of the clusters
print(kmeans_A.n_iter_)
#print the score of the clusters
print(kmeans_A.score(weather_A))
'''

weather_A.to_csv('weather_A.csv', index=False)

#plot automatically temperature and irradiation depending on the hour in 2 figures for type A in different colors depending on the cluster
#Temperature
plt.figure()
for i in range(0,len(weather_A.cluster)):
    plt.plot(weather_A[weather_A.cluster == i].Temp, '+')
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data')
#grid on
plt.grid(True)

#Irradiation
plt.figure()
for i in range(0,len(weather_A.cluster)):
    plt.plot(weather_A[weather_A.cluster == i].Irr, '+')
plt.xlabel('Hour')
plt.ylabel('Irradiation [W/m2]')
plt.title('Weather data')
#grid on
plt.grid(True)


plt.show()

'''
#Type A
#select the data for the buildings that are used and the temperature is below the cut off temperature
weather_A_ = weather[(weather.Temp < 16) & (weather.Irr > 0)]
#plot the data
plt.figure()
plt.plot(weather_A_.Temp)
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data type A')

#Type B
#select the data for the buildings that are not used and the temperature is above the cut off temperature
weather_B_ = weather[(weather.Temp > 16) & (weather.Irr > 0)]
#plot the data
plt.figure()
plt.plot(weather_B_.Temp)
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data type B')
#plt.show()

#separate the outliers from the data
#Type A
#calculate the mean and the standard deviation
mean_A = weather_A.Temp.mean()
std_A = weather_A.Temp.std()
#select the data that are outlier
outliers_A = weather_A[(weather_A.Temp < mean_A - 2*std_A) | (weather_A.Temp > mean_A + 2*std_A)]
#select the data that are not outlier
weather_A = weather_A[(weather_A.Temp > mean_A - 2*std_A) & (weather_A.Temp < mean_A + 2*std_A)]
#plot the data on the same figure with 2 colors
plt.figure()
plt.plot(weather_A.Temp)
plt.plot(outliers_A.Temp)
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
#plt.show()

#Type B
#calculate the mean and the standard deviation
mean_B = weather_B.Temp.mean()
std_B = weather_B.Temp.std()
#select the data that are outlier
outliers_B = weather_B[(weather_B.Temp < mean_B - 2*std_B) | (weather_B.Temp > mean_B + 2*std_B)]
#select the data that are not outlier
weather_B = weather_B[(weather_B.Temp > mean_B - 2*std_B) & (weather_B.Temp < mean_B + 2*std_B)]
#plot the data on the same figure with 2 colors
plt.plot(weather_B.Temp)
plt.plot(outliers_B.Temp)
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data')
plt.show()

#weather_A = weather_A[(weather_A.Temp > mean_A - 2*std_A) & (weather_A.Temp < mean_A + 2*std_A)]

'''


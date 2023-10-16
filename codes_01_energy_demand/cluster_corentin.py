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
#cluster the data in 4 clusters
kmeans_A = KMeans(n_clusters=6, random_state=0).fit(weather_A)
#add the cluster column to the dataframe
weather_A['Cluster'] = kmeans_A.labels_
#plot the data
plt.figure()
plt.scatter(weather_A.Temp, weather_A.Irr, c=weather_A.Cluster)
#plot the centroids in red
plt.scatter(kmeans_A.cluster_centers_[:, 0], kmeans_A.cluster_centers_[:, 1], c='red', s=200, alpha=0.5)
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

#plot temperature and irradiation depending on the hour in 2 figures for type A in 4 different colors depending on the cluster
plt.figure()
plt.plot(weather_A[weather_A.Cluster == 0].Temp, '+')
'''
plt.plot(weather_A[weather_A.Cluster == 1].Temp, '+')
plt.plot(weather_A[weather_A.Cluster == 2].Temp, '+')
plt.plot(weather_A[weather_A.Cluster == 3].Temp, '+')
plt.plot(weather_A[weather_A.Cluster == 4].Temp, '+')
plt.plot(weather_A[weather_A.Cluster == 5].Temp, '+')
'''
#plt.plot(weather_df.Temp)
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data')
#grid on
plt.grid(True)


plt.figure()
plt.plot(weather_A[weather_A.Cluster == 0].Irr, '+')
plt.plot(weather_A[weather_A.Cluster == 1].Irr, '+')
plt.plot(weather_A[weather_A.Cluster == 2].Irr, '+')
plt.plot(weather_A[weather_A.Cluster == 3].Irr, '+')
plt.plot(weather_A[weather_A.Cluster == 4].Irr, '+')
plt.plot(weather_A[weather_A.Cluster == 5].Irr, '+')
#plt.plot(weather_df.Temp)
plt.xlabel('Hour')
plt.ylabel('Irradiance [W/m2]')
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
    
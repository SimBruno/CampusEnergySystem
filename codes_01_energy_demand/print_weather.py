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

'''
#plot weather in 2 figures one for temperature and one for irradiation
plt.figure()
plt.plot(weather.Temp)
plt.xlabel('Hour')
plt.ylabel('Temperature [°C]')
plt.title('Weather data')

plt.figure()
plt.plot(weather.Irr)
plt.xlabel('Hour')
plt.ylabel('Irradiation [W/m2]')
plt.title('Weather data')
#plt.show()
'''

#cluster with sklearn using kmeans algorithm using 2 types: Type A) timesteps were the buildings are used (Monday to Friday 7am to 9pm) and the external temperature is below the cut off temperature of 16 °C.• Type B) the exact opposite of type A leaving the timesteps when the buildings are not used (night,weekend) and the external temperature is greater than 16 °C
#intialize the weather_df with columns names index, temperature, irradiation and type
weather_df = []

for k in range(0, 8759):
    #select only k that is between 7 and 21 modulo 24
    if (k % 24 >= 7 and k % 24 <= 21):
        if weather.Temp[k] > 16:
            weather_df.append([k, weather.Temp[k], weather.Irr[k], 'A'])
    else:
        weather_df.append([k, weather.Temp[k], weather.Irr[k], 'B'])

weather_df = pd.DataFrame(weather_df, columns=['Hours', 'Temp', 'Irr', 'Type'])

# Add outliers type
mean_w = weather_df.Temp.mean()
std_w = weather_df.Temp.std()
# Z-score = 3
ZSCORE = 2.3
weather_df.loc[(weather_df.Temp < mean_w - ZSCORE*std_w) | (weather_df.Temp > mean_w + ZSCORE*std_w), 'Type'] = 'O'

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
    
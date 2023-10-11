import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read in CSV file
df = pd.read_csv('/Users/simon/Documents/GitHub/MOES2023/report-group-3/codes_01_energy_demand/Weather.csv', header=0, encoding='unicode_escape')
df.columns = ['T_amb_degree_C', 'Irr_W_per_m2']
#add a column with the hours of the day starting at 0 and continuing until no more data
df['Hours'] = range(0, len(df.index))



# Plot temperature and irradiance vs hours of the day
plt.figure()
# Assuming you have an 'Hour' column in your DataFrame representing hours of the day
plt.plot(df.index, df['T_amb_degree_C'], label='Temperature (C)')
plt.plot(df.index, df['Irr_W_per_m2'], label='Irradiance (W/m2)')
plt.xlabel('Hours of the day')
plt.ylabel('Temperature (C) and Irradiance (W/m2)')
plt.title('Temperature and Irradiance vs Hours of the day')
plt.legend()
plt.show()

# Reshape the data for K-Means clustering of Irradiance vs hours of the day
X = df[['Hours','Irr_W_per_m2']]

# Perform K-Means clustering
kmeans_irr = KMeans(n_clusters=3, random_state=0).fit(X)

# Plot not in scatter way clustered Irradiance vs hours of the day



#test
plt.figure()
plt.scatter(df.index, df['Irr_W_per_m2'], c=kmeans_irr.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(kmeans_irr.cluster_centers_[:, 0], kmeans_irr.cluster_centers_[:, 1], c='red', s=50)
plt.xlabel('Hours of the day')
plt.ylabel('Irradiance [W/m2]')
plt.title('Irradiance vs Hours of the day (Clustered)')
plt.legend(['Hours (hr)', 'Irradiance (W/m2)'])
plt.show()


# Reshape the data for K-Means clustering of Temperature vs hours of the day
X = df[['T_amb_degree_C','Hours']]

# Perform K-Means clustering
kmeans_temp = KMeans(n_clusters=3, random_state=0).fit(X)

# Plot not in scatter way clustered Irradiance vs hours of the day



#test
plt.figure()
plt.scatter(df.index, df['T_amb_degree_C'], c=kmeans_temp.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(kmeans_temp.cluster_centers_[:, 0], kmeans_temp.cluster_centers_[:, 1], c='red', s=50)
plt.xlabel('Hours of the day')
plt.ylabel('Temperature [°C]')
plt.title('Temperature vs Hours of the day (Clustered)')
plt.legend(['Hours (hr)', 'Temperature (°C)'])
plt.show()

'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read in CSV file
df = pd.read_csv('/Users/simon/Documents/GitHub/MOES2023/report-group-3/codes_01_energy_demand/Weather.csv', header=0, encoding='unicode_escape')
df.columns = ['T_amb_degree_C', 'Irr_W_per_m2']

# Reshape the data for K-Means clustering
X = df[['T_amb_degree_C', 'Irr_W_per_m2']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Plot clustered Irradiance vs hours of the day using plot
plt.figure()

# Iterate through each cluster and plot separately
for cluster_label in range(3):
    cluster_data = df[kmeans.labels_ == cluster_label]
    plt.plot(cluster_data.index, cluster_data['Irr_W_per_m2'], label=f'Cluster {cluster_label}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50, label='Cluster Centers')
plt.xlabel('Hours of the day')
plt.ylabel('Irradiance [W/m2]')
plt.title('Irradiance vs Hours of the day (Clustered)')
plt.legend()
plt.show()


'''
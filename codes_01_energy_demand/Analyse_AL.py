import numpy as np; from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os


def load_data_weather_buildings():
    
    path = os.path.dirname(__file__) # the path to codes_01_energy_demand.py

    prop = pd.read_csv(os.path.join(path, "thermal_properties_copy.csv"),header=0,encoding = 'unicode_escape')
    prop.columns = ['Name', 'Area', 'Elec_heat', 'k_th', 'k_sun', 'people_gain']

    weather = pd.read_csv(os.path.join(path, "Weather.csv"),header=0,encoding = 'unicode_escape')
    weather.columns = ['Temp', 'Irr']

    buildings = pd.read_csv(os.path.join(path, "Buildings.csv"),header=0,encoding = 'unicode_escape')
    buildings.columns = ['Name', 'Year', 'Ground', 'Heat', 'Elec']



    return prop,weather,buildings



prop,weather,buildings=load_data_weather_buildings()



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
    return q_people


office_profile=occupancy_profile()[0]
class_profile=occupancy_profile()[1]
cantine_profile=occupancy_profile()[2]
elec_profile=occupancy_profile()[3]

def elec_gains(building_id: str,elec_profile):
    q_elec=elec_profile*buildings[buildings['Name']==building_id]['Elec'].to_numpy()/3654
    return q_elec

def Q_th_list(prop_id):
    T_cut=273+16
    T_int=273+21
    Q_th=np.zeros(8760)
    Q_th_25=np.zeros(8760)
    Q_th_50=np.zeros(8760)
    Q_th_75=np.zeros(8760)
    Q_diff=np.zeros(8760)
    Q_elec=elec_profile*buildings[buildings['Name']==prop_id]['Elec'].to_numpy()/3654
    Q_people=people_gains(office_profile,class_profile,cantine_profile)###.to_numpy()
    k_th=prop[prop['Name']==prop_id]['k_th'].to_numpy()[0]
    k_sun=prop[prop['Name']==prop_id]['k_sun'].to_numpy()[0]
    area=prop[prop['Name']==prop_id]['Area'].to_numpy()[0]
    T_ext=weather['Temp'].to_numpy()
    Irr=weather['Irr'].to_numpy()
    for i in range(8760):
        if T_ext[i]+273<=T_cut:
           Q_th[i]=elec_profile[i]*area*(k_th*(T_int-T_ext[i])-k_sun*Irr[i]*(1/1000)-Q_people[i])-Q_elec[i]*0.8 ###kWh
           Q_th_25[i]=elec_profile[i]*area*(k_th*(T_int-T_ext[i])-k_sun*1.25*Irr[i]*(1/1000)-Q_people[i])-Q_elec[i]*0.8 ###kWh
           Q_th_50[i]=elec_profile[i]*area*(k_th*(T_int-T_ext[i])-k_sun*1.50*Irr[i]*(1/1000)-Q_people[i])-Q_elec[i]*0.8 ###kWh
           Q_th_75[i]=elec_profile[i]*area*(k_th*(T_int-T_ext[i])-k_sun*1.75*Irr[i]*(1/1000)-Q_people[i])-Q_elec[i]*0.8 ###kWh
           Q_diff[i] = abs(Q_th[i] - Q_th_75[i]) 
        else:
           Q_th[i]=0
           Q_th_25[i]=0
           Q_th_50[i]=0
           Q_th_75[i]=0
           Q_diff[i] = 0

    return [Q_th,Q_th_25, Q_th_50,Q_th_75,Q_diff]

h=np.zeros(8760)
for i in range(8760):
    h[i]=i
    


Q=Q_th_list('BI') #### POUR Q_TH FAIRE Q[0]
#Q[1] = Q_thlist('BS')
#print(Q_th('BI'))
plt.figure()
plt.yscale("log")
plt.plot(h,Q[0],'b+', label="Reference case: k_sun")
#plt.plot(h,Q[1],'g.')
#plt.plot(h,Q[2],'r+')
plt.plot(h,Q[3],'y+', label="Increase of 75%: k_sun")
plt.xlabel('hours')
plt.ylabel('Qheating [kW]')
plt.title('Influence of ksun on Qheating')
plt.legend()
plt.figure()
plt.yscale("log")
plt.plot(h,Q[4],'b+')
plt.show()
#Test
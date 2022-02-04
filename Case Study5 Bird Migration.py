# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:08:46 2021

@author: kobienkung
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#________________________________________________________________________________________
birddata = pd.read_csv('bird_tracking.csv')
bird_names = pd.unique(birddata.bird_name)

for bird_name in bird_names:
    ix = birddata.bird_name == bird_name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    plt.plot(x, y, '.', label = bird_name)
plt.xlabel('Longtitude')
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.savefig('3traj.pdf')

#________________________________________________________________________________________
ix = birddata.bird_name == 'Eric'
speed = birddata.speed_2d[ix]
# np.isnan(speed) ~ check numerical?
# np.isnan(speed).any()
# np.sum(np.isnan(speed))
ind = np.isnan(speed)
plt.hist(speed[~ind], bins=np.linspace(0,30,20), density=True)
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')

speed.plot(kind='hist', range=[0,30])
# pd deal with NaNs automatically
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')
plt.show()

#________________________________________________________________________________________
# datetime.datetime.today()
timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
    (birddata.date_time[k][:-3], '%Y-%m-%d %H:%M:%S'))
        
'''
t1 = pd.Timestamp(birddata.date_time[0][:-3])
t1 = pd.to_datetime(birddata.date_time[0][:-3])
t2 = pd.Timestamp(birddata.date_time[1][:-3])
t2 = pd.Timestamp(birddata.date_time[1][:-3])
pd.to_timedelta(t2-t1, unit='day')
T3 = datetime.datetime.strptime(birddata.date_time[0][:-3], '%Y-%m-%d %H:%M:%S')
T4 = datetime.datetime.strptime(birddata.date_time[1][:-3], '%Y-%m-%d %H:%M:%S')
'''

birddata['timestamp'] = pd.Series(timestamps, index = birddata.index)

times = birddata.timestamp[birddata.bird_name == 'Eric']
elapsed_time = [time - times[0] for time in times]

# elapsed_time[1000] / datetime.timedelta(hours=1)
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)
plt.plot(elapsed_days)
plt.xlabel('Observation')
plt.ylabel('Elapsed time (day)')


nextday = 1
inds = []
daily_mean_speed = []
for i, t in enumerate(elapsed_days):
    if t < nextday:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(birddata.speed_2d[inds]))
        nextday += 1
        inds = []
        inds.append(i)

plt.plot(daily_mean_speed)
plt.xlabel('Day')
plt.ylabel('Mean speed (m/s)')

#________________________________________________________________________________________
proj = ccrs.Mercator()

plt.figure(figsize=(10,10))
ax = plt.axes(projection=proj)
ax.set_extent((-25, 20, 52, 10))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle = ':')

for name in bird_names:
    ix = birddata.bird_name == name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x, y, '.', transform = ccrs.Geodetic(), label = name)

plt.legend(loc='upper left')
plt.savefig('bird map.pdf')











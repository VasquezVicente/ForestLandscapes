import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#GLOBAL VARIABLES
frac = 0.02 # fraction of data used for each local regression (controls smoothness)
it=2
threshold_mid_point= 20  ## below 20% an event is concider dormant

data= pd.read_csv(r"timeseries/dataset_extracted/ceiba.csv")
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['dayYear'] = data['date'].dt.dayofyear
data['year']= data['date'].dt.year
data['date_num']= (data['date'] -data['date'].min()).dt.days

## plot all the timeseries of values
plt.figure(figsize=(12, 6))
for key, grp in data.groupby(['GlobalID']):
    plt.plot(grp['date_num'], grp['leafing'])
plt.xlabel('Days since start of study')
plt.ylabel('Leafing Index')
plt.title('Phenophase Detection')
plt.legend()
plt.show()

gid_unique= data['GlobalID'].unique()
all_df = []
for gid in gid_unique:
    indv = data[data['GlobalID'] == gid]
    full_date_range = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D') 
    full_df = pd.DataFrame({'date': full_date_range})
    full_df['dayYear'] = full_df['date'].dt.dayofyear
    full_df['date_num'] = (full_df['date'] - full_df['date'].min()).dt.days
    full_df['year'] = full_df['date'].dt.year
    indv = pd.merge(full_df, indv, on=['date', 'date_num', 'year', 'dayYear'], how='left')
    indv['leafing'] = indv['leafing'].interpolate(method='linear')
    indv['GlobalID'] = gid
    all_df.append(indv)

data = pd.concat(all_df, ignore_index=True)


all_points = []
for gid in gid_unique:
    print(f"Processing tree {gid}")
    data_gid= data[data['GlobalID'] == gid]
    data_gid['delta'] = data_gid['leafing'].diff() / data_gid['date_num'].diff()

    below = data_gid[data_gid['leafing'] < threshold_mid_point]
    below['group'] = (below['date_num'].diff() > 1).cumsum()
    mid_dates = below.groupby('group')['date_num'].median().values

    #for every mid point, get me the point that is not interpolated that is greater than 95% and its negativ
    if mid_dates.size == 0:
        print(f"No mid points found for tree {gid}, skipping...")
        continue

    for bkp in mid_dates:
        start_day = bkp - 120
        end_day = bkp + 120
        segment = data_gid[(data_gid['date_num'] >= start_day) & (data_gid['date_num'] <= end_day)]
        segment = segment[segment['leafing_predicted'].notna()]
        segment['seg_day'] = segment['date_num'] - bkp
        pre = segment[(segment['seg_day'] < 0) & (segment['leafing'] >= 90)]
        post = segment[(segment['seg_day'] > 0) & (segment['leafing'] >= 90)]
        print(f"bkp={bkp}, pre.shape={pre.shape}, post.shape={post.shape}, segment.shape={segment.shape}")
        if not pre.empty:
            t1 = pre.iloc[-1]
        else:
            print(f"No start leaf drop found for tree {gid} around mid point {bkp}")
            continue  # or handle accordingly
        if not post.empty:
            t2 = post.iloc[0]
        else:
            print(f"No end leaf drop found for tree {gid} around mid point {bkp}")
            continue  # or handle accordingly
        segment['drop_day'] = segment['date_num'] - t1['date_num']
        segment= segment[segment['seg_day'] < 0]
        segment['GlobalID'] = gid
        segment['mid_point'] = bkp
        all_points.append(segment)


all_points_df = pd.concat(all_points, ignore_index=True)
plt.figure(figsize=(12,  6))
plt.scatter(all_points_df['drop_day'], all_points_df['leafing'], marker='o')
plt.xlabel('Days relative to start of leaf drop')
plt.ylabel('Leafing Index')
plt.title('Phenophase Segments')
plt.legend()
plt.show()

from pygam import GAM, s
import numpy as np

# Prepare data
X = all_points_df['drop_day'].values.reshape(-1, 1)
y = all_points_df['leafing'].values

# Fit a GAM with a smooth spline term
gam = GAM(s(0, n_splines=10)).fit(X, y)

# Predict over a range for smooth plotting
xx = np.linspace(all_points_df['drop_day'].min(), all_points_df['drop_day'].max(), 200)
yy = gam.predict(xx)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(all_points_df['drop_day'], all_points_df['leafing'], marker='o', alpha=0.3, label='Observed')
plt.plot(xx, yy, color='red', linewidth=2, label='GAM fit')
plt.xlabel('Days relative to start of leaf drop')
plt.ylabel('Leafing Index')
plt.title('GAM Fit to Phenophase Segments')
plt.legend()
plt.show()









import pandas as pd
#we have 3 tree named  A, B, C
trees= ['A', 'B', 'C']
#tree A was observed in time= 2 an 4 with values 100 and 0
#tree B was observed in time= 1 and 3 with values 100 and 0
#tree C was observed in time= 2 and 3 with values 100 and 0
data = {
    'GlobalID': ['A', 'A', 'B', 'B', 'C', 'C'],
    'day': [2, 4, 1, 3, 2, 3],
    'leafing': [100, 0, 100, 0, 100, 0]
}

df = pd.DataFrame(data)
plt.figure(figsize=(12, 6))
plt.scatter(df['day'], df['leafing'], color='blue')
plt.show()

## we can calculate the rate of decay as the difference in leafing divided by the difference in days
decay= df.groupby('GlobalID').apply(lambda group: (group['leafing'].diff() / group['day'].diff()).iloc[-1])
print(decay)

# here we know  that the correct rate of decay is -100 because we have that prior but due to the sampling we are mislead

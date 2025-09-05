import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#GLOBAL VARIABLES
threshold_mid_point= 20  ## below 20% an event is concider dormant


######################below here is simulated data##########################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#species called platypoidium petandra has a true leaf dropped pattern defined as below:
t=range(10,60,1)
y = [100]*20 + [95, 85, 65, 40, 20, 0] + [0]*24
df_true=pd.DataFrame({'time': t, 'leafing': y})

plt.figure(figsize=(12, 6))
plt.plot(df_true['time'], df_true['leafing'], label='True Leaf Drop Pattern')
plt.show()

# the range of leafing values is:
leafing_values= range(0, 100, 1)
#we have trees named:
trees= ['A', 'B', 'C','D','E','F','G','H']
# observed across 7 years
years= range(2015, 2022,1)
# we have categories l=0 for trees labeled by humans with associated error e
# we have categories l=1 for trees labeled by the model with associated error e=3 when 0<= y <=20 or 0<=y<=100
labeled= ['human', 'model']
observed_values= []
for year in years:
    t_values = np.random.choice(t, size=2, replace=False)
    for tree in trees:
        for t_val in t_values:
            label= np.random.choice(labeled, p=[0.2, 0.8])
            if label == 'human':
                leafing = df_true[df_true['time'] == t_val]['leafing'].values[0]
                error = 0
            else:
                true_leafing = df_true[df_true['time'] == t_val]['leafing'].values[0]
                if true_leafing <= 20 or true_leafing >= 80:
                    error = 3
                else:
                    error = 30
                low = max(0, true_leafing - error)
                high = min(100, true_leafing + error)
                leafing = np.random.uniform(low, high)

            observed_values.append({
                'tree': tree,
                'year': year,
                't_val': t_val,
                'leafing': leafing,
                'label': label,
                'error': error
            })

df_observed = pd.DataFrame(observed_values)

plt.figure(figsize=(12, 6))
plt.scatter(df_observed['t_val'], df_observed['leafing'], c='blue', label='Observed Data', alpha=0.6)
plt.show()

#we can fit a logistic curve to the data
def logistic(t, a, b):
    return 100 / (1 + np.exp(a * (t - b)))

# Fit species-level curve
popt, pcov = curve_fit(logistic, df_observed['t_val'], df_observed['leafing'], p0=[0.5, 10])
a, b = popt
print(f"a = {a:.3f}, b = {b:.3f}")

df_observed['leafing_predicted'] = logistic(df_observed['t_val'], *popt)

plt.figure(figsize=(12, 6))
plt.scatter(df_observed['t_val'], df_observed['leafing'], label='Observed Data')
plt.plot(df_observed.sort_values(by='t_val')['t_val'], df_observed.sort_values(by='t_val')['leafing_predicted'], label='Fitted Curve', color='red')
plt.xlabel('Day')
plt.ylabel('Leafing')
plt.title('Leafing Over Time with Fitted Curve')
plt.legend()
plt.show()

##########################above here is simulated data####################################

##########################below here is real data####################################

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
    indv['interpolated']= np.where(indv['leafing'].isna(), True, False)
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



# Fit species-level curve
popt, pcov = curve_fit(logistic, all_points_df['drop_day'], all_points_df['leafing'], p0=[0.5, 10])
a, b = popt
print(f"a = {a:.3f}, b = {b:.3f}")

all_points_df['leafing_predicted'] = logistic(all_points_df['drop_day'], *popt)

plt.figure(figsize=(12, 6))
plt.scatter(all_points_df['drop_day'], all_points_df['leafing'], label='Observed Data')
plt.plot(all_points_df.sort_values(by='drop_day')['drop_day'], all_points_df.sort_values(by='drop_day')['leafing_predicted'], label='Fitted Curve', color='red')
plt.xlabel('Day')
plt.ylabel('Leafing')
plt.title('Leafing Over Time with Fitted Curve')
plt.legend()
plt.show()



##########################above here is real data####################################

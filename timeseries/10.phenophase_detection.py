import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ruptures as rpt

#GLOBAL VARIABLES
threshold_mid_point= 20  ## below 20% an event is concider dormant

##FUNCTIONS
#we can fit a logistic curve to the data
def logistic(t, a, b):
    return 100 / (1 + np.exp(a * (t - b)))

######################below here is simulated data##########################
###true is simulated here
start_date = pd.Timestamp('2018-04-04')
end_date = pd.Timestamp('2024-03-18')
t = pd.date_range(start=start_date, end=end_date, freq='D')
t_year= range(1, 367)
y= [100]*20 + [95, 85, 65, 40, 20, 10,0] + [0]*20 + [5, 25, 45, 70, 90, 100] + [100]*313
y_shifted_10= [100]*30 + [95, 85, 65, 40, 20, 10,0] + [0]*20 + [5, 25, 45, 70, 90, 100] + [100]*303

df_true=pd.DataFrame({'time': t_year, 'leafing': y})
df_true_shifted_10=pd.DataFrame({'time': t_year, 'leafing': y_shifted_10})
#all years true pattern
full_true = []
for t_value in t:
    day_of_year = t_value.dayofyear
    if t_value.year == 2021:
        true_leafing = df_true_shifted_10.loc[
            df_true_shifted_10['time'] == day_of_year, 'leafing'
        ].values[0]
    else:
        true_leafing = df_true.loc[
            df_true['time'] == day_of_year, 'leafing'
        ].values[0]
    
    full_true.append({'date': t_value, 'leafing': true_leafing})
df_true_all_years = pd.DataFrame(full_true)
df_true_all_years['dayYear'] = df_true_all_years['date'].dt.dayofyear
df_true_all_years['year']= df_true_all_years['date'].dt.year


plt.figure(figsize=(12, 6))
for year in df_true_all_years['year'].unique():
    subset = df_true_all_years[df_true_all_years['year'] == year]
    plt.plot(subset['dayYear'], subset['leafing'], label=f'Year {year}')
plt.legend()
plt.xlim(5, 80)
plt.show()

plt.plot(df_true_all_years['date'], df_true_all_years['leafing'], label='Leaf Drop Pattern (with 2021 shifted)')
plt.legend()
plt.show()
##############

#species called platypoidium petandra has a true leaf dropped pattern defined as below and it is completely synchronous across all individuals.

# Generate t_samples with random monthly interv
df_true_all_years['dayYear'] = df_true_all_years['date'].dt.dayofyearals

t_samples = [start_date]
while t_samples[-1] < end_date:
    #interval = np.random.randint(25, 36) # Randomly choose the next interval between 25 a
    interval = np.random.randint(13, 17) # Randomly choose the next interval between 13 and 17 days ~ biweekly data
    interval = np.random.randint(6, 9)   # Randomly choose the next interval between 6 and 7 days ~ weekly data
    next_date = t_samples[-1] + pd.Timedelta(days=interval)
    if next_date > end_date:
        break
    t_samples.append(next_date)

t_samples = pd.to_datetime(t_samples)
print(len(t_samples), t_samples)
# Use t_samples for simulated observations

data= pd.read_csv(r"timeseries/dataset_extracted/cavallinesia.csv")
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['dayYear'] = data['date'].dt.dayofyear
data['year']= data['date'].dt.year
data['date_num']= (data['date'] -data['date'].min()).dt.days

t_samples = data['date'].unique()

for t in t_samples:
    print(t)
#now every tree in the set:
trees= ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
#is sampled at every t_samples dy
#with leafing values in the set:
leafing_values= range(0, 100, 1)         #we have Leave Cover Percentage (LFP) values from 0 to 100
#and the leafing values are determined by humans or model with different accuracies
labeled= ['human', 'model']                     # LFP - determined by humans or model
pheno_years= range(start_date.year, end_date.year + 1, 1)                      #  we observed across 7 years


preferred_shifts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0}
preferred_shifts_year= {2018: 0, 2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0}
shifts_keyed = {}
for pheno_year in pheno_years:
        shifts_keyed[pheno_year] = {}
        shift_pheno_year = preferred_shifts_year[pheno_year]
        shifts_keyed[pheno_year]['year_shift'] = shift_pheno_year
        print(f"    Shift for pheno year {pheno_year}: {shift_pheno_year}")

        for tree in trees:
            # Sample around preferred shift for each tree
            shift_individual = int(np.round(np.random.normal(loc=preferred_shifts[tree], scale=0)))
            shifts_keyed[pheno_year][tree] = shift_individual
            print(f"        Shift for tree {tree}: {shift_individual}")


observed_values= []
for t_sample in t_samples:
    print(f"Sampling date: {t_sample.date()}")
    # Get leafing values 15 days before and after t_sample
    mask = (df_true_all_years['date'] >= t_sample - pd.Timedelta(days=15)) & (df_true_all_years['date'] <= t_sample + pd.Timedelta(days=15))
    leafing_window = df_true_all_years.loc[mask, 'leafing'].values
    population_leafing= df_true_all_years[df_true_all_years['date']== t_sample]['leafing'].values[0]

    #whats the year of the t_sample
    pheno_year = t_sample.year
    print(f"  Pheno year: {pheno_year}")

    for tree in trees:
        # get the shifts for year and tree
        year_shift = shifts_keyed[pheno_year]['year_shift']
        tree_shift = shifts_keyed[pheno_year][tree]
        #print(f"    Year shift: {year_shift}, Tree shift: {tree_shift}")
        total_shift = year_shift + tree_shift
        print(f"    Total shift for tree {tree} in year {pheno_year}: {total_shift}")
        #now i need to look up the value of leafing at population_leafing + total_shift
        try:
            leafing_value = leafing_window[15 + total_shift]
        except IndexError:
            print(f"    Warning: Shifted index {15 + total_shift} is out of bounds for leafing_window.")
            leafing_value = leafing_window[15]  #15 is the index of the population mean in the window

        # now was the value observed by human or model
        label= np.random.choice(labeled, p=[0.2, 0.8])
        if label == 'human':
            leafing = leafing_value
            error = 0 # humans are perfect
        else:
            if leafing_value <= 20 or leafing_value >= 80:
                error = 3  # model is very accurate at the extremes
            else:
                error = 30 # model is less accurate in the middle range
            low = max(0, leafing_value - error)
            high = min(100, leafing_value + error)

            leafing = np.random.uniform(low, high)
            #print(f"low={low}, high={high}, leafing={leafing}")

        #print(f"Tree: {tree}, Date: {t_sample.date()}, Actual Leafing: {leafing_value}, Label: {label}, Observed Leafing: {leafing:.2f}, Error: {error}")
        observed_values.append({
            'tree': tree,
            'date': t_sample.date(),
            'actual_leafing': leafing_value,
            'label': label,
            'observed_leafing': leafing,
            'error': error
        })
        
df_observed = pd.DataFrame(observed_values)

df_observed['date'] = pd.to_datetime(df_observed['date'])
df_observed['date_num'] = (df_observed['date'] - df_observed['date'].min()).dt.days
df_observed['dayYear'] = df_observed['date'].dt.dayofyear
df_observed['year']= df_observed['date'].dt.year
# Color by different tree
plt.figure(figsize=(12, 6))
for tree in df_observed['tree'].unique():
        subset = df_observed[df_observed['tree'] == tree]
        plt.plot(subset['date'], subset['observed_leafing'], label=f'Tree {tree}')
#plt.scatter(df_true['time'], df_true['leafing'], color='black', label='True Pattern', zorder=5)
plt.plot(df_true_all_years['date'], df_true_all_years['leafing'], color='black', label='True Pattern', zorder=5)
plt.xlabel('Day')
plt.ylabel('Leafing')
plt.title('Observed Data Colored by Tree')
plt.show()

plt.figure(figsize=(12, 6))
for year in df_true_all_years['year'].unique():
    subset = df_true_all_years[df_true_all_years['year'] == year]
    plt.plot(subset['dayYear'], subset['leafing'], label=f'Year {year}')
    subset_obs = df_observed[df_observed['year'] == year]
    plt.scatter(subset_obs['dayYear'], subset_obs['observed_leafing'], label=f'Observed {year}', alpha=0.3)
plt.legend()
plt.xlim(1, 100)
plt.show()



df_observed.to_csv("timeseries/simulated_phenophase_data.csv", index=False)


all_events = []

for tree in trees:
    indv = df_observed[df_observed['tree'] == tree]

    # (1) fill timeline
    full_date_range = pd.date_range(df_observed['date'].min(), df_observed['date'].max(), freq='D')
    full_df = pd.DataFrame({'date': full_date_range})
    full_df['dayYear'] = full_df['date'].dt.dayofyear
    full_df['date_num'] = (full_df['date'] - full_df['date'].min()).dt.days
    full_df['year'] = full_df['date'].dt.year
    indv = pd.merge(full_df, indv, on=['date','date_num'], how='left')
    indv['leafing'] = indv['observed_leafing'].interpolate(method='linear')

    # (2) mask of below-threshold days
    indv['below'] = indv['leafing'] < 60
    mask = indv['below'].values
    dates = indv.loc[mask, 'date'].values
    date_nums = indv.loc[mask, 'date_num'].values

    if len(dates) == 0:
        continue  # no events for this tree

    # (3) group consecutive days
    group_ids = np.diff(date_nums, prepend=date_nums[0]) > 1
    groups = group_ids.cumsum()

    # (4) summarize each group
    for g in np.unique(groups):
        g_dates = dates[groups == g]
        start_date = g_dates[0]
        end_date = g_dates[-1]
        duration = (end_date - start_date).astype('timedelta64[D]').item()
        median_date = g_dates[len(g_dates) // 2]

        all_events.append({
            'tree': tree,
            'event_id': g,
            'start_date': start_date,
            'end_date': end_date,
            'duration': duration,
            'median_date': median_date
        })


all_events_df= pd.DataFrame(all_events)
all_events_df['duration'].mean()

#the year is year and tval is day of year transform to date

#you could simply detect the mid point as the median of all points below the threshold_mid_point
all_points = []
for tree in trees:
    print(f"Processing tree {tree}")
    data_gid = df_observed[df_observed['tree'] == tree]
    below = data_gid[data_gid['leafing'] < threshold_mid_point]
    mid_dates = below.groupby('year')['date_num'].median().values
    print(f"mid_dates for tree {tree}: {mid_dates}")
    for bkp in mid_dates:
        print(f"bkp={bkp}")
        start_day = bkp - 40
        end_day = bkp + 1
        segment = data_gid[(data_gid['date_num'] >= start_day) & (data_gid['date_num'] <= end_day)]
        segment = segment[segment['t_val']<=200]
        segment['segmentid'] = f"{tree}_{bkp}"
        print(segment)
        all_points.append(segment)


all_points_df = pd.concat(all_points, ignore_index=True)

plt.figure(figsize=(12, 6))
for segm in all_points_df['segmentid'].unique():
    subset = all_points_df[all_points_df['segmentid'] == segm]
    plt.plot(subset['t_val'], subset['leafing'], label=f'Segment {segm}', marker='o')

# Overlay the true pattern, centered at 0
plt.plot(df_true['time'], df_true['leafing'], color='black', linewidth=3, label='True Pattern')

plt.xlabel('Days relative to start of leaf drop')
plt.ylabel('Leafing Index')
plt.title('Phenophase Segments - Daily Sampling')
plt.xlim(1, 50)
plt.show()

# now that you have aligned all the points to the start of leaf drop you could determine the rate of decay
# dif of leafing / dif of days for every tree in each year and get the average k value to see if we approach the real value
k_values = []
for segm in all_points_df['segmentid'].unique():
    subset = all_points_df[all_points_df['segmentid'] == segm]
    subset['delta'] = subset['leafing'].diff() / subset['date_num'].diff()
    k_hat = subset[(subset['delta'] < 0)& (subset['delta']<-0.5)]['delta']
    if not k_hat.empty:
        k_values.append(k_hat.mean())


print("Average k value across all trees and years:", np.mean(k_values))
print("Actual k value from true pattern:", k_hat_by_segment)




# Fit species-level curve
popt, pcov = curve_fit(logistic, all_points_df['t_val'], all_points_df['leafing'], p0=[0.5, 10])
a, b = popt
print(f"a = {a:.3f}, b = {b:.3f}")

all_points_df['leafing_predicted'] = logistic(all_points_df['t_val'], *popt)

full_range= range(all_points_df['t_val'].min(), all_points_df['t_val'].max()+1)
full_range_df = pd.DataFrame({'t_val': full_range})
full_range_df['leafing_predicted'] = logistic(full_range_df['t_val'], *popt)

plt.figure(figsize=(12, 6))
plt.scatter(all_points_df['t_val'], all_points_df['leafing'], label='Observed Data')
plt.plot(full_range_df['t_val'], full_range_df['leafing_predicted'], label='Fitted Curve', color='red')
plt.plot(df_true['time'], df_true['leafing'], color='black', linewidth=3, label='True Pattern')
plt.xlabel('Day')
plt.ylabel('Leafing')
plt.title('Leafing Over Time with Fitted Curve')
plt.legend()
plt.xlim(1, 50)
plt.show()


## test with all trees
for segm in all_points_df['segmentid'].unique():
    subset = all_points_df[all_points_df['segmentid'] == segm]
    full_dates = pd.date_range(start=subset['date'].min(), end=subset['date'].max(), freq='D')
    full_df = pd.DataFrame({'date': full_dates})
    full_df = pd.merge(full_df, subset, on='date', how='left')
    full_df['t_val'] = full_df['date'].dt.dayofyear

    full_df['leafing_predicted'] = logistic(full_df['t_val'], *popt)
    full_df['leafing'].fillna(full_df['leafing_predicted'], inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(full_df['t_val'], full_df['leafing_predicted'], label=f'Segment {segm}', marker='o')
    plt.plot(df_true['time'], df_true['leafing'], color='black', linewidth=3, label='True Pattern')
    plt.xlabel('Days relative to start of leaf drop')
    plt.ylabel('Leafing Index')
    plt.title(f'Phenophase Segment and Fitted Curve - {segm}')
    plt.xlim(1, 50)
    plt.legend()
    plt.show()


##########################above here is simulated data####################################

##########################below here is real data####################################


data_weekly= data[data['date']>= "2022-08-01"]
data_monthly= data[data['date']<= "2022-08-01"]


## plot all the timeseries of values
plt.figure(figsize=(12, 6))
for key, grp in data_weekly.groupby(['GlobalID']):
    plt.plot(grp['date_num'], grp['leafing'])
plt.xlabel('Days since start of study')
plt.ylabel('Leafing Index')
plt.title('Phenophase Detection')
plt.legend()
plt.show()




#### now we can detect the mid point as the median of all points below the threshold_mid_point
all_points = []

for gid in data_weekly['GlobalID'].unique():
    print(f"Processing tree {gid}")
    data_gid= data_weekly[data_weekly['GlobalID'] == gid]

    below = data_gid[data_gid['leafing'] < threshold_mid_point]
    below['group'] = (below['date_num'].diff() > 60).cumsum()

    mid_dates = below.groupby('group')['date_num'].apply(lambda x: (x.max() - x.min()) / 2 + x.min())
    if mid_dates.size == 0:  #check in case no mid points were found
        print(f"No mid points found for tree {gid}, skipping...")
        continue

    for bkp in mid_dates:
        print(f"bkp={bkp}")
        start_day = bkp - 140
        end_day = bkp + 100
        segment = data_gid[(data_gid['date_num'] >= start_day) & (data_gid['date_num'] <= end_day)]
        segment['seg_day'] = segment['date_num'] - bkp
        segment['seg_id'] = f"{gid}_{bkp}" 
        segment['group'] = below[below['date_num'].between(start_day, end_day)]['group'].iloc[0]
        all_points.append(segment)

all_points_df = pd.concat(all_points, ignore_index=True)
all_points_df= all_points_df[all_points_df['group']==0] 

plt.figure(figsize=(12, 6))
for segm in all_points_df['seg_id'].unique()[0:10]:  #limit to first 10 segments for clarity
    subset = all_points_df[all_points_df['seg_id'] == segm]
    plt.plot(subset['seg_day'], subset['leafing'], label=f'Segment {segm}', marker='o')
plt.xlabel('Days since start of study')
plt.ylabel('Leafing Index')
plt.title('Phenophase Segments')
plt.show()





all_points_df_below_0 = all_points_df[all_points_df['seg_day'] < 10]


all_shifted= []
for segm in all_points_df_below_0['seg_id'].unique():
    subset = all_points_df_below_0[all_points_df_below_0['seg_id'] == segm]

    if subset.empty:
        continue
    has_above = (subset['leafing'] >= 50).any()
    has_below = (subset['leafing'] < 50).any()

    if not (has_above and has_below):
        print(f"  Skipping {segm}, no crossing at 50.")
        continue

    # last value above 50
    above = subset[subset['leafing'] >= 50].iloc[-1]['seg_day']
    # first value below 50
    below = subset[subset['leafing'] < 50].iloc[0]['seg_day']

    # midpoint between them (optional, you can keep just above/below)
    day_50 = (above + below) / 2

    print(f"  Segment {segm}: crosses 50% between {above} and {below}, midpoint ≈ {day_50}")

    #now shift the seg_day to have day_50 as 0
    subset['drop_day'] = subset['seg_day'] - day_50
    all_shifted.append(subset)

all_shifted_df = pd.concat(all_shifted, ignore_index=True)
all_shifted_df['abs_day'] = all_shifted_df['drop_day'] - all_shifted_df['drop_day'].min().round()
#for every segment plot it with its seg id
plt.figure(figsize=(12, 6))
for segm in all_shifted_df['seg_id'].unique()[0:10]:  #limit to first 10 segments for clarity
    subset = all_shifted_df[all_shifted_df['seg_id'] == segm]
    plt.plot(subset['abs_day'], subset['leafing'], label=f'Segment {segm}', marker='o')
plt.xlabel('Days since start of study')
plt.ylabel('Leafing Index')
plt.title('Phenophase Segments')
plt.show()




popt, pcov = curve_fit(logistic, all_shifted_df['abs_day'], all_shifted_df['leafing'], p0=[0.5, 10])
a, b = popt
print(f"a = {a:.3f}, b = {b:.3f}")
full_range= range(0,253,1)
full_range_df = pd.DataFrame({'abs_day': full_range})
full_range_df['leafing_predicted'] = logistic(full_range_df['abs_day'], *popt)

#for every segment plot it with its seg id
plt.figure(figsize=(12, 6))
for segm in all_shifted_df['seg_id'].unique()[0:100]:  #limit to first 10 segments for clarity
    subset = all_shifted_df[all_shifted_df['seg_id'] == segm]
    plt.scatter(subset['abs_day'], subset['leafing'], label=f'Segment {segm}', marker='o')
plt.plot(full_range_df['abs_day'], full_range_df['leafing_predicted'], label='Fitted Curve', color='red')
plt.xlabel('Days since start of study')
plt.ylabel('Leafing Index')
plt.title('Phenophase Segments')
plt.show()



for segm in all_shifted_df['seg_id'].unique()[0:10]:  #limit to first 10 segments for clarity
    subset = all_shifted_df[all_shifted_df['seg_id'] == segm]
    full_range = range(int(subset['abs_day'].min()), int(subset['abs_day'].max()) + 1)
    full_range_df = pd.DataFrame({'abs_day': full_range})

    full_merged_df = pd.merge(full_range_df, subset, on='abs_day', how='left')
    full_merged_df['leafing_predicted'] = logistic(full_merged_df['abs_day'], *popt)
    full_merged_df['leafing'].fillna(full_merged_df['leafing_predicted'], inplace=True)
    
    full_range_df['leafing_predicted'] = logistic(full_range_df['abs_day'], *popt)

     # Plotting

    plt.figure(figsize=(12, 6))
    plt.scatter(subset['abs_day'], subset['leafing'], label=f'Segment {segm}', marker='o')
    plt.plot(full_merged_df['abs_day'], full_merged_df['leafing'], label='Fitted Curve', color='red')
    plt.xlabel('Days since start of study')
    plt.ylabel('Leafing Index')
    plt.title(f'Phenophase Segment {segm}')
    plt.show()


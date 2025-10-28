import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


######################below here is simulated data##########################
###true is simulated here
start_date = pd.Timestamp('2018-04-04')
end_date = pd.Timestamp('2024-03-18')
t = pd.date_range(start=start_date, end=end_date, freq='D')
t_year= range(1, 367)

y= [100]*40 + [95, 85, 65, 40, 20, 10,0] + [0]*20 + [5, 25, 45, 70, 90, 100] + [100]*293
y_shifted_e= [100]*45 + [95, 85, 65, 40, 20, 10,0] + [0]*20 + [5, 25, 45, 70, 90, 100] + [100]*288

y_shifted_10= [100]*50 + [95, 85, 65, 40, 20, 10,0] + [0]*20 + [5, 25, 45, 70, 90, 100] + [100]*283
y_shifted_e_2021= [100]*55 + [95, 85, 65, 40, 20, 10,0] + [0]*20 + [5, 25, 45, 70, 90, 100] + [100]*278


trees= ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
years= range(start_date.year, end_date.year + 1, 1)                      #  we observed across 7 years

df_true=pd.DataFrame({'time': t_year, 'leafing': y})
df_true_shifted_10=pd.DataFrame({'time': t_year, 'leafing': y_shifted_10})
df_true_shifted_e=pd.DataFrame({'time': t_year, 'leafing': y_shifted_e})
df_true_shifted_e_2021=pd.DataFrame({'time': t_year, 'leafing': y_shifted_e_2021})
#all years true pattern
full_true = []
for t_value in t:
    day_of_year = t_value.dayofyear
    year= t_value.year
    for tree in trees:
            if year == 2021 and tree == 'E':
                true_leafing = df_true_shifted_e_2021.loc[
                    df_true_shifted_e_2021['time'] == day_of_year, 'leafing'
                ].values[0]
            elif year == 2021 and tree != 'E':
                true_leafing = df_true_shifted_10.loc[
                    df_true_shifted_10['time'] == day_of_year, 'leafing'
                ].values[0]
            elif year != 2021 and tree == 'E':
                true_leafing = df_true_shifted_e.loc[
                    df_true_shifted_e['time'] == day_of_year, 'leafing'
                ].values[0]
            else:
                true_leafing = df_true.loc[
                    df_true['time'] == day_of_year, 'leafing'
                ].values[0]
            full_true.append({'date': t_value, 'leafing': true_leafing, 'tree': tree})
df_true_all_years = pd.DataFrame(full_true)
df_true_all_years['dayYear'] = df_true_all_years['date'].dt.dayofyear
df_true_all_years['year']= df_true_all_years['date'].dt.year

df_true_all_years[df_true_all_years['tree']=='E']

plt.figure(figsize=(12, 6))
for year in df_true_all_years['year'].unique():
    for tree in trees:
        subset = df_true_all_years[(df_true_all_years['year'] == year) & (df_true_all_years['tree'] == tree)]
        plt.plot(subset['dayYear'], subset['leafing'])
plt.legend()
plt.xlim(5, 150)
plt.show()


##############

#species called platypoidium petandra has a true leaf dropped pattern defined as below and it is completely synchronous across all individuals.

# Use t_samples for simulated observations
data= pd.read_csv(r"timeseries/dataset_extracted/cavallinesia.csv")
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['dayYear'] = data['date'].dt.dayofyear
data['year']= data['date'].dt.year
data['date_num']= (data['date'] -data['date'].min()).dt.days
# after you prepare data['date'], dayYear, year, etc.


t_samples= data['date'].unique()
# Prepare list to store new dates
new_dates = []

# Loop through each year
existing_dates= set(pd.to_datetime(data['date']))
candidates= pd.date_range(start=start_date, end=end_date, freq='D')
candidates = [d for d in candidates if d not in existing_dates]

n_new = min(106, len(candidates))
# Sample new dates
if n_new > 0:
        sampled = np.random.choice(candidates, size=n_new, replace=False)
        sampled = pd.to_datetime(sampled)
        new_dates.extend(sampled)


df_new_dates = pd.DataFrame({'date': new_dates})
data_augmented = pd.concat([data, df_new_dates], ignore_index=True)
t_samples= data_augmented['date'].unique()
#now every tree in the set:
trees= ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
#is sampled at every t_samples dy
#with leafing values in the set:
leafing_values= range(0, 100, 1)         #we have Leave Cover Percentage (LFP) values from 0 to 100
#and the leafing values are determined by humans or model with different accuracies
labeled= ['human', 'model']                     # LFP - determined by humans or model
pheno_years= range(start_date.year, end_date.year + 1, 1)                      #  we observed across 7 years


observed_values= []
for t_sample in t_samples:
    print(f"Sampling date: {t_sample.date()}")
    pheno_year = t_sample.year
    print(f"  Pheno year: {pheno_year}")

    for tree in trees:
        # get the shifts for year and tree
        leafing_value = df_true_all_years[
            (df_true_all_years['date'] == t_sample) & (df_true_all_years['tree'] == tree)
            ]['leafing'].values[0]
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
for year in df_true_all_years['year'].unique():
    for tree in trees:
        subset = df_true_all_years[(df_true_all_years['year'] == year) & (df_true_all_years['tree'] == tree)]
        plt.plot(subset['dayYear'], subset['leafing'])
        subset_obs = df_observed[(df_observed['year'] == year) & (df_observed['tree'] == tree)]
        plt.scatter(subset_obs['dayYear'], subset_obs['observed_leafing'], label=f'Observed {tree} {year}', alpha=0.6)

plt.xlim(20, 120)
plt.show()



df_observed.to_csv("timeseries/simulated_phenophase_data.csv", index=False)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import pickle
import ruptures as rpt
import statsmodels.api as sm
from scipy.signal import savgol_filter
import seaborn as sns

data= pd.read_csv(r"timeseries/dataset_predictions/hura_sgbt.csv")

with open(r'timeseries/models/xgb_model.pkl', 'rb') as file:
      model = pickle.load(file)

X=data[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]

Y= data[['area', 'score', 'tag', 'GlobalID', 'iou',
       'date', 'latin', 'polygon_id']]
X_predicted=model.predict(X)
df_final = Y.copy()  # Copy Y to keep the same structure
df_final['leafing_predicted'] = X_predicted

df_final['date'] = pd.to_datetime(df_final['date'], format='%Y_%m_%d')
df_final['dayYear'] = df_final['date'].dt.dayofyear
df_final['year']= df_final['date'].dt.year
df_final['date_num']= (df_final['date'] -df_final['date'].min()).dt.days

globalID= df_final['GlobalID'].unique()
len(globalID)
plt.figure(figsize=(20, 6))
for tree in df_final['GlobalID'].unique():
    indv = df_final[df_final['GlobalID'] == tree]
    tagn= indv['tag'].unique()[0]
    plt.plot(indv['date'], indv['leafing_predicted'], label=f"Tree {tagn}", alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Hura Crepitans: Leaf coverage vs Date')
plt.legend(title="ForestGeo Tag", bbox_to_anchor=(1, 1), loc='upper left')
plt.show()


df_final['tag_numeric'] = pd.Categorical(df_final['tag']).codes
lowess = sm.nonparametric.lowess(df_final['leafing_predicted'], df_final['dayYear'], frac=0.09)  # Adjust frac for smoothing level
plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2, label="LOWESS Trend")
plt.scatter(df_final['dayYear'], df_final['leafing_predicted'], c=df_final['tag_numeric'], cmap='viridis', alpha=0.7)

plt.xlabel('Day of Year')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Leafing Predicted vs Day of Year with Trend Line')
plt.show()


all_df = []
for tree in globalID:
    indv = df_final[df_final['GlobalID'] == tree]
    full_date_range = pd.date_range(start=df_final['date'].min(), end=df_final['date'].max(), freq='D')
    full_df = pd.DataFrame({'date': full_date_range})
    full_df['dayYear'] = full_df['date'].dt.dayofyear
    full_df['date_num'] = (full_df['date'] - full_df['date'].min()).dt.days
    full_df['year'] = full_df['date'].dt.year
    indv = pd.merge(full_df, indv, on=['date', 'date_num', 'year', 'dayYear'], how='left')

    indv['leafing_predicted'] = indv['leafing_predicted'].interpolate(method='linear')
    lowess_smoothed = sm.nonparametric.lowess(indv['leafing_predicted'], indv['date_num'], frac=0.1)

    #plt.figure(figsize=(12, 6))
    #plt.scatter(indv['date_num'], indv['leafing_predicted'], alpha=0.5, label='Raw Data')
    #plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], color='orange', linewidth=2, label='LOWESS Smoothed')
    #plt.xlabel('Day of Year')
    #plt.ylabel('Leafing Predicted')
    #plt.grid(True)
    #plt.title('Leafing Predicted vs. Day of Year (LOWESS Smoothed)')
    #plt.legend()
    #plt.show()

    signal = indv['leafing_predicted'].values
    algo_python = rpt.Pelt(model="rbf",jump=1,min_size=15).fit(signal)
    penalty_value = 30
    bkps_python = algo_python.predict(pen=penalty_value)


    #plt.figure(figsize=(12, 6))
    #plt.scatter(indv['date_num'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
    #plt.xlabel('Day of year')
    ##plt.ylabel('Leafing Predicted')
    ##plt.grid(True)
    #plt.title('Leafing Predicted vs Day of Year')
    #plt.legend()
    #plt.colorbar(label='Year')
    #for bp in bkps_python:
    #    plt.axvline(x=bp, color='red', linestyle='--', linewidth=2)
    #plt.show()

    indv['breakpoint'] = False
    indv.loc[indv['date_num'].isin(bkps_python), 'breakpoint'] = True
    indv['GlobalID'] = tree
    all_df.append(indv)
    print("one done")

out_df = pd.concat(all_df, ignore_index=True)
out_df['GlobalID']

break_group= out_df[out_df['breakpoint']==True]

def calculate_slope(x_values, y_values):
    slope = (y_values[-1] - y_values[0]) / (x_values[-1] - x_values[0])
    return slope

all_classified=[]
for tree in globalID:
    indv= out_df[out_df['GlobalID']==tree]
    bkps_python=indv[indv['breakpoint']==True]['date_num'].values
    
    for index, row in indv.iterrows():
        if row['breakpoint'] == True:
            breakpoint_day = row['date_num']
            pre_start = breakpoint_day - 10
            pre_end = breakpoint_day
            post_start = breakpoint_day
            post_end = breakpoint_day + 10
            if pre_start >= 0 and post_end < len(indv):
                    pre_data = indv.loc[(indv['date_num'] >= pre_start) & (indv['date_num'] < pre_end)]
                    post_data = indv.loc[(indv['date_num'] > post_start) & (indv['date_num'] <= post_end)]
                    
                    # Calculate the slopes
                    pre_slope = calculate_slope(pre_data['date_num'].values, pre_data['leafing_predicted'].values)
                    post_slope = calculate_slope(post_data['date_num'].values, post_data['leafing_predicted'].values)
                    
                    # Classify the breakpoint based on the slopes
                    if pre_slope > 0 and post_slope >0:                        # they are both positive obiously increasing
                        # Leaf flush (pre-slope positive, post-slope positive but smaller)
                        indv.loc[index, 'break_type'] = 'end_leaf_flush'
                        print('end_leaf_flush',breakpoint_day)
                    elif pre_slope < 0 and post_slope<0 :                       #both negative obiously decreasing
                        # Leaf drop (pre-slope negative, post-slope even more negative)
                        indv.loc[index, 'break_type'] = 'start_leaf_drop'
                        print('leaf_drop', breakpoint_day, pre_slope,post_slope)
                    elif pre_slope > 0 and -0.25 <= post_slope <= 0:
                        # End of leaf flush (pre-slope positive, post-slope within the range of -0.25 to 0)
                        indv.loc[index, 'break_type'] = 'end_leaf_flush'
                        print('leaf_flush', breakpoint_day, pre_slope, post_slope)
                    elif pre_slope > 0 and post_slope < 0 and post_slope < -0.25:
                        # End of leaf flush (pre-slope positive, post-slope negative but above -0.25)
                        indv.loc[index, 'break_type'] = 'start_leaf_drop'
                        print('leaf_flush', breakpoint_day, pre_slope, post_slope)
    all_classified.append(indv)

out_df_2 = pd.concat(all_classified, ignore_index=True)


plt.figure(figsize=(12, 6))
    plt.scatter(indv['date_num'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
    plt.xlabel('Day of year')
    plt.ylabel('Leafing Predicted')
    plt.grid(True)
    plt.title('Leafing Predicted vs Day of Year')
    plt.legend()
    plt.colorbar(label='Year')
    for bp in bkps_python:
        plt.axvline(x=bp, color='red', linestyle='--', linewidth=2)
    plt.show()
# I am goint to have to extract the species specific metric
# maybe i can do that with the etire dataset, then filter the trees that do not match the pattern(noise)
# i am going to have to aggregate the dates

df = df_final.groupby('date', as_index=False)['leafing_predicted'].mean()
full_date_range = pd.date_range(start=df_final['date'].min(), end=df_final['date'].max(), freq='D')
full_df = pd.DataFrame({'date': full_date_range})
full_df['dayYear'] = full_df['date'].dt.dayofyear
full_df['date_num'] = (full_df['date'] - full_df['date'].min()).dt.days
full_df['year'] = full_df['date'].dt.year
full = pd.merge(full_df, df, on=['date'], how='left')
full['leafing_predicted'] = full['leafing_predicted'].interpolate(method='linear')

plt.scatter(full['date'], full['leafing_predicted'])
plt.show()

signal = full['leafing_predicted'].values
algo_python = rpt.Pelt(model="rbf",jump=1,min_size=15).fit(signal)
penalty_value = 30
bkps_python = algo_python.predict(pen=penalty_value)


plt.figure(figsize=(12, 6))
plt.scatter(full['date_num'], full['leafing_predicted'], c=full['year'], cmap='viridis', label='Leafing Predicted')
plt.xlabel('Date numeric')
plt.ylabel('Mean leaf coverage')
plt.grid(True)
plt.title('Hura crepitans mean leaf coverage vs date')
plt.legend()
plt.colorbar(label='Year')

for bp in bkps_python:
    plt.axvline(x=bp, color='red', linestyle='--', linewidth=2)
    date_label = full.loc[full['date_num'] == bp, 'date'].dt.date.values  # Extract only date part
    if len(date_label) > 0:
        plt.text(bp, full['leafing_predicted'].mean(), str(date_label[0]), 
                 rotation=45, color='black', fontsize=10, verticalalignment='center_baseline')
plt.show()


breakouts_2019= break_group[break_group['year']==2019]

plt.figure(figsize=(10, 6))
sns.histplot(breakouts_2019['dayYear'], kde=True, bins=365, color='blue', stat='density')
plt.title('Distribution of Breakouts in 2019 (Day of Year)', fontsize=16)
plt.xlabel('Day of Year', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
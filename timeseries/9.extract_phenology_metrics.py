import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import pickle
import ruptures as rpt
import statsmodels.api as sm
from scipy.signal import savgol_filter
import seaborn as sns

data= pd.read_csv(r"timeseries/dataset_extracted/prioria.csv")

data['latin']
##date wrangling
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['dayYear'] = data['date'].dt.dayofyear
data['year']= data['date'].dt.year
data['date_num']= (data['date'] -data['date'].min()).dt.days


##plot all trees leafing prediction against the date
globalID= data['GlobalID'].unique()
len(globalID)
plt.figure(figsize=(20, 6))
for tree in data['GlobalID'].unique():
    indv = data[data['GlobalID'] == tree]
    tagn= indv['tag'].unique()[0]
    plt.plot(indv['date'], indv['leafing'], label=f"Tree {tagn}", alpha=0.7)
    
plt.xlabel('Date')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Hura Crepitans: Leaf coverage vs Date')
plt.legend(title="ForestGeo Tag", bbox_to_anchor=(1, 1), loc='upper left')
plt.show()


# SPECIES SPECIFIC PATTERN YEARLY
data['tag_numeric'] = pd.Categorical(data['tag']).codes
lowess = sm.nonparametric.lowess(data['leafing_predicted'], data['dayYear'], frac=0.09)  # Adjust frac for smoothing level
plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2, label="LOWESS Trend")
plt.scatter(data['dayYear'], data['leafing_predicted'], c=data['tag_numeric'], cmap='viridis', alpha=0.7)
plt.xlabel('Day of Year')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Leafing Predicted vs Day of Year with Trend Line')
plt.show()


all_df = []
for tree in globalID:
    indv = data[data['GlobalID'] == tree]
    full_date_range = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
    full_df = pd.DataFrame({'date': full_date_range})
    full_df['dayYear'] = full_df['date'].dt.dayofyear
    full_df['date_num'] = (full_df['date'] - full_df['date'].min()).dt.days
    full_df['year'] = full_df['date'].dt.year
    indv = pd.merge(full_df, indv, on=['date', 'date_num', 'year', 'dayYear'], how='left')
    indv['leafing'] = indv['leafing'].interpolate(method='linear')
    #lowess_smoothed = sm.nonparametric.lowess(indv['leafing_predicted'], indv['date_num'], frac=0.1)
    #plt.figure(figsize=(12, 6))
    #plt.scatter(indv['date_num'], indv['leafing_predicted'], alpha=0.5, label='Raw Data')
    #plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], color='orange', linewidth=2, label='LOWESS Smoothed')
    #plt.xlabel('Day of Year')
    #plt.ylabel('Leafing Predicted')
    #plt.grid(True)
    #plt.title('Leafing Predicted vs. Day of Year (LOWESS Smoothed)')
    #plt.legend()
    #plt.show()
    signal = indv['leafing'].values
    algo_python = rpt.Pelt(model="rbf",jump=1,min_size=15).fit(signal)
    penalty_value = 25
    bkps_python = algo_python.predict(pen=penalty_value)
    plt.figure(figsize=(12, 6))
    plt.scatter(indv['date_num'], indv['leafing'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
    plt.xlabel('Day of year')
    plt.ylabel('Leafing Predicted')
    plt.grid(True)
    plt.title('Leafing Predicted vs Day of Year')
    plt.legend()
    plt.colorbar(label='Year')
    for bp in bkps_python:
        plt.axvline(x=bp, color='red', linestyle='--', linewidth=2)
    plt.show()
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
                    pre_slope = calculate_slope(pre_data['date_num'].values, pre_data['leafing'].values)
                    post_slope = calculate_slope(post_data['date_num'].values, post_data['leafing'].values)
                    
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
len(out_df_2['GlobalID'].unique())

out_df_2.to_csv(r'timeseries/dataset_analysis/quararibea_analysis.csv')


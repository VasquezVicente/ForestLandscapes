import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import pickle
import ruptures as rpt
import statsmodels.api as sm


data= pd.read_csv(r"timeseries/dataset_predictions/dipteryx_sgbt.csv")

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
    signal = indv['leafing_predicted'].values
    algo_python = rpt.Pelt(model="rbf", jump=1, min_size=5).fit(signal)
    penalty_value = 30
    bkps_python = algo_python.predict(pen=penalty_value)
    indv['breakpoint'] = False
    indv.loc[indv['date_num'].isin(bkps_python), 'breakpoint'] = True
    indv['GlobalID'] = tree
    all_df.append(indv)

out_df = pd.concat(all_df, ignore_index=True)

break_group= out_df[out_df['breakpoint']==True]


plt.figure(figsize=(10, 6))
plt.hist(break_group['dayYear'], bins=365, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Day of the Year')
plt.ylabel('Breakpoint Count')
plt.title('Concentration of Breakpoints by Day of the Year')
plt.grid(True)
plt.show()



globalID= all['GlobalID'].unique()

indv= all[all['GlobalID']=='badee35d-7b09-4d87-9b4a-b052af7c3749']



for tree in globalID:
     indv= all[all['GlobalID']==tree].reset_index()
     indv['leafing_predicted'] = indv['leafing_predicted'].interpolate(method='linear')

     

plt.figure(figsize=(12, 6))
plt.scatter(indv.dropna(subset=['leafing_predicted'])['date_num'],
            indv.dropna(subset=['leafing_predicted'])['leafing_predicted'],
            c=indv.dropna(subset=['leafing_predicted'])['year'], cmap='viridis')
plt.xlabel('Day of year')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Leafing Predicted vs Day of Year')
plt.legend()
plt.colorbar(label='Year')
plt.show()








# Fit LOWESS smoothing
lowess = sm.nonparametric.lowess(indv['leafing_predicted'], indv['date_num'], frac=0.1)  # frac controls smoothness

# Plot scatter and LOWESS trend
plt.figure(figsize=(12, 6))
plt.scatter(indv['date_num'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted', alpha=0.6)
plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2, label='LOWESS Trend')

plt.xlabel('Day of year')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Leafing Predicted vs Day of Year with LOWESS Trend Line')
plt.legend()
plt.colorbar(label='Year')
plt.show()


# Extract the signal
#for year 2019

signal = indv['leafing_predicted'].values  # Get the leafing predictions
len(signal)
# Apply rupture's dynamic programming method
algo_python = rpt.Pelt(model="rbf", jump=1, min_size=5).fit(
    signal
)
penalty_value=30
bkps_python = algo_python.predict(pen=penalty_value)



# Plot the leafing predictions
plt.figure(figsize=(12, 6))
plt.scatter(indv['date_num'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
plt.xlabel('Day of year')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Leafing Predicted vs Day of Year')
plt.legend()
plt.colorbar(label='Year')

# Add the detected breakpoints as dashed vertical lines
for bp in bkps_python:
    plt.axvline(x=bp, color='red', linestyle='--', linewidth=2)

# Show the plot
plt.show()


indv['breakpoint'] = False  
indv.loc[indv['date_num'].isin(bkps_python), 'breakpoint'] = True

indv[indv['breakpoint']==True]
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import pickle
import ruptures as rpt
import statsmodels.api as sm


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

    #plt.figure(figsize=(12, 6))
    #plt.scatter(indv['dayYear'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
    #plt.xlabel('Day of year')
    #plt.ylabel('Leafing Predicted')
    #plt.grid(True)
    #plt.title('Leafing Predicted vs Day of Year')
    #plt.legend()
    #plt.colorbar(label='Year')
    #plt.show()

    signal = indv['leafing_predicted'].values
    algo_python = rpt.Pelt(model="rbf",jump=1,min_size=15).fit(signal)
    penalty_value = 30
    bkps_python = algo_python.predict(pen=penalty_value)


    #plt.figure(figsize=(12, 6))
    #plt.scatter(indv['date_num'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
    #plt.xlabel('Day of year')
    #plt.ylabel('Leafing Predicted')
    #plt.grid(True)
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

break_group= out_df[out_df['breakpoint']==True]
break_group[break_group['dayYear'].isna()]

plt.scatter(break_group['year'], break_group['dayYear'])
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
bkps_python
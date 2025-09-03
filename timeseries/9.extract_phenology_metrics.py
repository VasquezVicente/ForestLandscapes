import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import pickle
import ruptures as rpt
import statsmodels.api as sm
from scipy.signal import savgol_filter
import seaborn as sns
from timeseries.utils import create_consensus_polygon
from timeseries.utils import create_overlap_density_map
import os
import geopandas as gpd

path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"                 ## path to the data folder
path_crowns=os.path.join(path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")  ## location of the timeseries of polygons
crowns=gpd.read_file(path_crowns)                                                      ## read the file using geopandas
species_subset= crowns[crowns['latin']=='Ceiba pentandra'].reset_index()         ## geodataframe to be used as template to extract features

unique_globalids = species_subset['GlobalID'].unique()
all_gid_polygons = []
for gid in unique_globalids:
    subset = species_subset[species_subset['GlobalID'] == gid]
    grid_size = 0.1
    x_coords, y_coords, density_matrix = create_overlap_density_map(subset, grid_size=grid_size)
    consensus_polygon = create_consensus_polygon(x_coords, y_coords, density_matrix)

    for idx, row in subset.iterrows():
            poly = row['geometry']
            distance = poly.hausdorff_distance(consensus_polygon)
            subset.loc[idx, 'hausdorff_distance'] = distance
            print(f"Polygon has Hausdorff distance: {distance:.2f}")
    
    distances = subset['hausdorff_distance']
    mean_distance = distances.mean()
    std_distance = distances.std()
    all_gid_polygons.append(subset)

    #this now plots correctly aligned
    # fig, ax = plt.subplots(figsize=(10, 10))
    # extent = [subset.total_bounds[0], subset.total_bounds[2], subset.total_bounds[1], subset.total_bounds[3]]
    # im = ax.imshow(density_matrix, extent=extent, origin='lower', cmap='hot', alpha=0.7)
    # subset['geometry'].plot(ax=ax, edgecolor='white', facecolor='none', alpha=0.8, linewidth=1)
    # #plot concensus polygon if it exists
    # if consensus_polygon is not None:
    #     gpd.GeoSeries(consensus_polygon).plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2)
    # plt.colorbar(im, ax=ax, label='Overlap Count')
    # plt.title(f"Density Map with Polygons - {gid}")
    # plt.suptitle(f"Mean Hausdorff Distance: {mean_distance:.2f}, Std Dev: {std_distance:.2f}")
    # plt.show()

all_polygons= pd.concat(all_gid_polygons, ignore_index=True)

sumary=all_polygons.groupby('GlobalID').agg({'hausdorff_distance': ['mean', 'std', 'min', 'max', 'count']})

plt.figure(figsize=(12, 8))
# Encode GlobalID as numeric for coloring
all_polygons['gid_numeric'] = pd.Categorical(all_polygons['GlobalID']).codes
scatter = plt.scatter(all_polygons['date'], all_polygons['hausdorff_distance'], c=all_polygons['gid_numeric'], cmap='tab20', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Hausdorff Distance to Consensus Polygon')
plt.title('Hausdorff Distance Over Time for Ceiba pentandra')
plt.grid(True)
plt.colorbar(scatter, label='GlobalID (encoded)')
plt.show()



gid_th= sumary[sumary[('hausdorff_distance', 'mean')] < 4].index.tolist()  ## we are using 10 as threshold for now
len(gid_th)

data= pd.read_csv(r"timeseries/dataset_extracted/ceiba.csv")
data= data[data['GlobalID'].isin(gid_th)]  ## filter to only those with good polygon representation

##date wrangling
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['dayYear'] = data['date'].dt.dayofyear
data['year']= data['date'].dt.year
data['date_num']= (data['date'] -data['date'].min()).dt.days


##plot all trees leafing prediction against the date
globalID= data['GlobalID'].unique()
len(globalID)
plt.figure(figsize=(20, 6))

# Encode GlobalID as numeric for coloring
for gid in globalID:
    indv = data[data['GlobalID'] == gid]
    plt.plot(indv['date'], indv['leafing_predicted'], label=str(gid))
plt.xlabel('Date')
plt.ylabel('Leafing Predicted')
plt.title('Ceiba pentandra: Leaf coverage vs Date')
#plt.legend(title='GlobalID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
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





counter = 1  # Initialize counter
all_df = []
for tree in globalID:
    indv = data[data['GlobalID'] == tree]
    # indv['dayYear'] = indv['date'].dt.dayofyear
    # indv['date_num'] = (indv['date'] - indv['date'].min()).dt.days
    # indv['year'] = indv['date'].dt.year
    full_date_range = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
    full_df = pd.DataFrame({'date': full_date_range})
    full_df['dayYear'] = full_df['date'].dt.dayofyear
    full_df['date_num'] = (full_df['date'] - full_df['date'].min()).dt.days
    full_df['year'] = full_df['date'].dt.year
    indv = pd.merge(full_df, indv, on=['date', 'date_num', 'year', 'dayYear'], how='left')
    indv['leafing'] = indv['leafing'].interpolate(method='linear')

    ##breakpoint detection
    signal = indv['leafing'].values
    algo_python = rpt.Pelt(model="rbf",jump=1,min_size=10).fit(signal)
    penalty_value = 25 ## 35 for hura
    bkps_python = algo_python.predict(pen=penalty_value)

    # plt.figure(figsize=(12, 6))
    # plt.scatter(indv['date_num'], indv['leafing'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
    # plt.xlabel('Day of year')
    # plt.ylabel('Leafing Predicted')
    # plt.grid(True)
    # plt.title(tree)
    # plt.legend()
    # plt.colorbar(label='Year')
    # for bp in bkps_python:
    #     plt.axvline(x=bp, color='red', linestyle='--', linewidth=2)
    # plt.show()
    print(tree)
    indv['breakpoint'] = False
    indv.loc[indv['date_num'].isin(bkps_python), 'breakpoint'] = True
    indv['GlobalID'] = tree
    all_df.append(indv)
    print("one done")



out_df = pd.concat(all_df, ignore_index=True)
out_df['GlobalID']


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

out_df_2.to_csv(r'timeseries/dataset_analysis/hura_analysis.csv')



##############################################################
selected_trees=['b0e0daf6-3d04-4565-ad3f-42b83e21c188', 'b5133434-1910-4069-9e34-899fac89dea3']
data= pd.read_csv(r"timeseries/dataset_extracted/ceiba.csv")
#data= data[data['GlobalID'].isin(selected_trees)]  ## filter to only those with good polygon representation
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['dayYear'] = data['date'].dt.dayofyear
data['year']= data['date'].dt.year
data['date_num']= (data['date'] -data['date'].min()).dt.days

all_trees=data['GlobalID'].unique()  # Replace with actual GlobalIDs
all_df = []
for gid in all_trees:
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

test = pd.concat(all_df, ignore_index=True)

plt.figure(figsize=(12, 6))
for gid in all_trees:
    indv = test[test['GlobalID'] == gid]
    plt.plot(indv['date_num'], indv['leafing'], label=str(gid))
plt.show()


# i want all days in test that are below leafing 20
dict={}
for gid in all_trees:
    indv = test[test['GlobalID'] == gid]
    below_20 = indv[indv['leafing'] < 15]
    below_20['group'] = (below_20['date_num'].diff() > 1).cumsum()
    mid_dates = below_20.groupby('group')['date_num'].median().values
    dict[gid] = mid_dates
    print(gid, mid_dates)





plt.figure(figsize=(12, 6))
for gid in all_trees[0:5]:
    mid_dates = dict[gid]
    for bkp in mid_dates:
        start_day = bkp - 60
        end_day = bkp + 60
        segment = test[(test['GlobalID'] == gid) & (test['date_num'] >= start_day) & (test['date_num'] <= end_day)]
        segment = segment[segment['leafing_predicted'].notna()]
        segment['seg_day'] = segment['date_num'] - bkp
        plt.plot(segment['seg_day'], segment['leafing'], marker='o')
        #plt.axvline(x=0, color='red', linestyle='--', label='Breakpoint')
plt.title('Segments around Breakpoints')
plt.xlabel('Date Num')
plt.ylabel('Leafing')
plt.legend()
plt.show()

for bkp in mid_dates:
    start_day = bkp - 60
    end_day = bkp + 60
    segment = test[(test['date_num'] >= start_day) & (test['date_num'] <= end_day)]
    segment['seg_day'] = segment['date_num'] - bkp
    plt.plot(segment['seg_day'], segment['leafing'], marker='o')
    #plt.axvline(x=0, color='red', linestyle='--', label='Breakpoint')
plt.title('Segments around Breakpoints')
plt.xlabel('Date Num')
plt.ylabel('Leafing')
plt.legend()
plt.show()

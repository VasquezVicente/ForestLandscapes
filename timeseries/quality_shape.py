import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union
##CODE FOR EXPLORING THE SHAPE AND SIZE OF THE CROWNS
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"                 ## path to the data folder
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")                         ## orthomosaics locally aligned location 
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")  ## location of the timeseries of polygons
crowns=gpd.read_file(path_crowns)                                                      ## read the file using geopandas
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")       ## polygon ID defines the identity of tree plus date it was taken
species_subset= crowns[crowns['latin']=='Alseis blackiana'].reset_index()         ## geodataframe to be used as template to extract features


x=5
globalIDS = species_subset['GlobalID'].unique()  # Get unique GlobalID
indv= species_subset[species_subset['GlobalID'] == globalIDS[x]]  # Select the first GlobalID
indv['GlobalID']
indv['area'] = indv['geometry'].area  # Calculate area of each polygon

plt.figure(figsize=(10, 6))
plt.plot(indv['date'], indv['iou'], marker='o', linestyle='-', color='b', label='Area')
plt.title(f"Area of {globalIDS[x]} Over Time")
plt.xlabel("Date")
plt.ylabel("Area")
plt.legend()
plt.grid()
plt.show()

mean_iou = indv['iou'].mean()
print(f"Mean Area for {globalIDS[x]}: {mean_iou:.2f}")



def create_overlap_density_map(gdf, grid_size=0.5):
    """Create a density map showing polygon overlap intensity
    Parameters:
        gdf (GeoDataFrame): GeoDataFrame containing the polygons.
        grid_size (float): Size of the grid cells for density calculation.
    """
    # Get bounds of all geometries
    bounds = gdf.total_bounds
    x_min, y_min, x_max, y_max = bounds
    
    # Create grid
    x_coords = np.arange(x_min, x_max, grid_size)
    y_coords = np.arange(y_min, y_max, grid_size)
    
    # Initialize density matrix
    density_matrix = np.zeros((len(y_coords), len(x_coords)))
    
    # For each grid point, count how many polygons contain it
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            point = Point(x, y)
            count = sum(geom.contains(point) for geom in gdf['geometry'])
            density_matrix[i, j] = count
    
    return x_coords, y_coords, density_matrix

# Create density map
grid_size = 0.3
x_coords, y_coords, density_matrix = create_overlap_density_map(indv, grid_size=grid_size)
density_matrix = (density_matrix - np.min(density_matrix)) / (np.max(density_matrix) - np.min(density_matrix))
density_matrix[density_matrix < 0.4] = np.nan
polygons_from_mask = []

for i in range(density_matrix.shape[0]):
    for j in range(density_matrix.shape[1]):
        if not np.isnan(density_matrix[i, j]):  # If not masked (>0.5)
            # Get grid cell coordinates
            x = x_coords[j]
            y = y_coords[i]
            
            # Create square polygon for this grid cell
            square = Polygon([
                (x, y),
                (x + grid_size, y),
                (x + grid_size, y + grid_size),
                (x, y + grid_size)
            ])
            polygons_from_mask.append(square)
# Union all squares into single polygon
if polygons_from_mask:
    density_polygon = unary_union(polygons_from_mask)
else:
    density_polygon = None

# Visualize
fig, ax = plt.subplots(figsize=(10, 10))
indv['geometry'].plot(ax=ax, edgecolor='gray', facecolor='none', alpha=0.3)
if density_polygon:
    gpd.GeoSeries([density_polygon]).plot(ax=ax, color='red', alpha=0.7)
plt.title("Density Consensus Polygon")
plt.show()

dates= []
mean_densities = []
for idx, row in indv.iterrows():
    current_polygon = row['geometry']
    current_date = row['date']

    # Extract density values within current polygon
    polygon_density_values = []
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            point = Point(x, y)
            if current_polygon.contains(point):
                polygon_density_values.append(density_matrix[i, j])
    
    # Calculate statistics for this polygon's density values
    if polygon_density_values:
        mean_density = np.nanmean(polygon_density_values)
        max_density = np.nanmax(polygon_density_values)
        print(f"Date: {current_date}, Mean density: {mean_density:.3f}, Max density: {max_density:.3f}")
        dates.append(current_date)
        mean_densities.append(mean_density)
    










# Get the reference polygon (the good one)
reference_polygon = density_polygon

iou_scores = []
dates = []

for idx, row in indv.iterrows():
    current_polygon = row['geometry']
    current_date = row['date']

    # extract pixels from density_matrix
    density_matrix
    
    # Calculate intersection and union
    intersection = reference_polygon.intersection(current_polygon)
    union = reference_polygon.union(current_polygon)
    
    # Calculate IoU
    iou = intersection.area / union.area if union.area > 0 else 0
    
    iou_scores.append(iou)
    dates.append(current_date)
    
    print(f"Date: {current_date}, IoU: {iou:.4f}")

# Add IoU scores to the dataframe
indv_with_iou = indv.copy()
indv_with_iou['iou_score'] = iou_scores

import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# Convert dates to numerical values (days since minimum date)
date_series = pd.to_datetime(dates, format='%Y_%m_%d')
min_date = date_series.min()
date_nums = (date_series - min_date).days

# Calculate LOWESS trend
lowess_result = lowess(iou_scores, date_nums, frac=0.5)
trend_line = lowess_result[:, 1]

plt.figure(figsize=(12, 6))
plt.plot(dates, iou_scores, marker='o', linestyle='-', color='r', label='IoU Score')
plt.plot(dates, trend_line, '--', color='blue', alpha=0.8, linewidth=2, label='LOWESS Trend')
plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Perfect Match (IoU=1)')
plt.title(f"IoU Scores relative to reference polygon - {globalIDS[x]}")
plt.xlabel("Date")
plt.ylabel("IoU Score")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


residuals = np.array(iou_scores) - trend_line
std_threshold = 2.0  # 2 standard deviations
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

# Method 2: Interquartile range approach (more robust)
q75, q25 = np.percentile(residuals, [75, 25])
iqr = q75 - q25
outlier_threshold = 1.5 * iqr
outlier_mask_iqr = (residuals < (q25 - outlier_threshold)) | (residuals > (q75 + outlier_threshold))


outlier_mask = outlier_mask_iqr

# Add outlier classification to dataframe
indv_with_iou['residual'] = residuals
indv_with_iou['is_outlier'] = outlier_mask

# Trend line
plt.figure(figsize=(12, 6))
plt.plot(dates, trend_line, '--', color='green', linewidth=2, label='LOWESS Trend')
plt.plot(indv_with_iou['date'], indv_with_iou['iou_score'], marker='o', linestyle='-', color=indv_with_iou['is_outlier'].map({True: 'red', False: 'blue'}), label='IoU Score')
# Confidence bands

plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
plt.title(f"IoU Outlier Detection - {globalIDS[x]}")
plt.xlabel("Date")
plt.ylabel("IoU Score")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



def buffered_consensus(gdf, buffer_distance=-0.5):
    """Create consensus by buffering union inward"""
    if len(gdf) == 0:
        return None
    # Union all polygons
    union_geom = gdf['geometry'].unary_union
    # Buffer inward to get core area
    consensus = union_geom.buffer(buffer_distance)
    return consensus



indv_corrected = indv_with_iou.copy()

# Replace their geometry with buffered consensus
if density_polygon is not None and not density_polygon.is_empty:
    indv_corrected.loc[outlier_mask, 'geometry'] = density_polygon
    print(f"Replaced {outlier_mask.sum()} outlier polygons with consensus geometry")
else:
    print("No valid consensus geometry to use for replacement")


from timeseries.utils import generate_leafing_pdf

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
out_pdf = r"//stri-sm01/ForestLandscapes/UAVSHARE/BCI_50ha_timeseries/videos/alseis/alseis_leafing4.pdf"

generate_leafing_pdf(indv_corrected, out_pdf, path_ortho, crowns_per_page=12, variables=['date', 'iou_score'])


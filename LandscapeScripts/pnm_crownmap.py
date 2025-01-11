import os
import geopandas as gpd
import rasterio
from matplotlib import pyplot as plt
import numpy as np
import Metashape
import open3d as o3d
import laspy
# we need to combine the point clouds
list_tiles_main=os.listdir(r"\\stri-sm01\ForestLandscapes\LandscapeRaw\ALS\Airborne Lidar Panama 2023\05. Parque Metropolitano\03. LAZ Classified")
path_tiles_main=r"\\stri-sm01\ForestLandscapes\LandscapeRaw\ALS\Airborne Lidar Panama 2023\05. Parque Metropolitano\03. LAZ Classified"
list_tiles=[f for f in list_tiles_main if f.endswith(".laz")]
all_points = []
all_colors = []

# Iterate over the list of tiles and process each tile
for i in range(len(list_tiles)):
    tile_path = os.path.join(path_tiles_main, list_tiles[i])
    tile_cloud = laspy.read(tile_path)
    points = np.vstack((tile_cloud.x, tile_cloud.y, tile_cloud.z, tile_cloud.red, tile_cloud.green, tile_cloud.blue, tile_cloud.classification)).T
    points_noground = points[points[:, 6] != 2]   # Remove ground points ==2
    points_max_color_value = np.max(points_noground[:, 3:6])  # Get the max color value for normalization of color values
    points_noground[:, 3:6] = points_noground[:, 3:6] / points_max_color_value  # Normalize color values for it to fit into open3d
    
    # Append points and colors to the lists
    all_points.append(points_noground[:, 0:3])
    all_colors.append(points_noground[:, 3:6])
    print('Tile', i+1, 'processed')

# Combine all points and colors into a single array
all_points = np.vstack(all_points)
all_colors = np.vstack(all_colors)

# Create a single Open3D point cloud with the combined data
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(all_points)
combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

# Optionally, visualize the combined point cloud
o3d.visualization.draw_geometries([combined_pcd])

#write the point cloud to a file
o3d.io.write_point_cloud(r"D:\PNM_2023.ply", combined_pcd)





def create_surface_model(cloud_path, output_path):
    doc=Metashape.Document()
    doc.save(path = "D:/projecttemp.psx")
    chunk=doc.addChunk()
    chunk.importPointCloud(cloud_path,format=Metashape.PointCloudFormatPLY,crs=Metashape.CoordinateSystem("EPSG::32617"))
    doc.save()
    proj = Metashape.OrthoProjection()
    proj.crs=Metashape.CoordinateSystem("EPSG::32617")
    chunk.buildDem(source_data=Metashape.PointCloudData,interpolation=Metashape.Extrapolated,projection=proj)
    if chunk.elevation:
            chunk.exportRaster(output_path, source_data = Metashape.ElevationData,projection= proj)


create_surface_model(r"D:\PNM_2023.ply", r"D:\PNM_2023_dsm1.tif")
#open the lidar ascii file



ant=r"D:\20241220_pnmmetrop_m3e_dsm.tif"
PNM_lidar=r"D:\PNM_2023_dsm1.tif"

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

# Open the source rasters
with rasterio.open(ant) as src:
    ant_data = src.read(1)
    ant_meta = src.meta

with rasterio.open(PNM_lidar) as src:
    PNM_data = src.read(1)
    PNM_meta = src.meta

# Create a destination array for the resampled data
resampled_PNM_data = np.empty_like(ant_data)

# Resample PNM_lidar to match ant_data
reproject(
    source=PNM_data,
    destination=resampled_PNM_data,
    src_transform=PNM_meta['transform'],
    src_crs=PNM_meta['crs'],
    dst_transform=ant_meta['transform'],
    dst_crs=ant_meta['crs'],
    resampling=Resampling.bilinear
)

# Calculate the delta raster
delta_raster = ant_data - resampled_PNM_data

# Optionally, save the delta raster to a new file

with rasterio.open("delta_raster.tif", "w", **delta_meta) as dst:
    dst.write(delta_raster.astype(np.float32), 1)

delta_raster= np.where(delta_raster>50, np.nan, delta_raster)
delta_raster= np.where(delta_raster<-50, np.nan, delta_raster)


plt.figure(figsize=(10, 10))       
plt.imshow(delta_raster, cmap='coolwarm', vmin=-50, vmax=50)
plt.colorbar()
plt.show()



path=r"D:\Arboles_PNM_export1\PNM_2024_crownmap.shp"
species=r"D:\BCI_50ha\aux_files\ListadoEspecies.csv"

# Load the shapefile
gdf = gpd.read_file(path)

gdf.columns

gdf= gdf[['GlobalID','CreationDa', 'Creator', 'EditDate',
       'Editor', 'Flowering', 'Tag', 'Category', 'Life_form', 'Notes',
       'Status', 'Observer', 'Species', 'Lianas', 'illuminati', 'Crown',
       'Dead_stand', 'New_leaves', 'Senecent_l', 'Fruiting', 'leafing',
       'Latin', 'geometry']]


path_crown=r"C:\Users\VasquezV\Downloads\panama_lianas_centroids_final\panama_lianas_centroids_final.shp"
gdf_crown = gpd.read_file(path_crown)


for row in gdf_crown.iterrows():
    print(row[1]['geometry'])
    nearest_geom=nearest_points(row[1]['geometry'], gdf_crown['geometry'])
    print(nearest_geom)

# Calculate the distance to the nearest point for each point
gdf_crown['nearest_distance'] = gdf_crown.apply(calculate_nearest, other_gdf=gdf_crown, axis=1)

# Print the GeoDataFrame with the new column
print(gdf_crown[['geometry', 'nearest_distance']])

#calculate the centroid of the crowns
gdf_crown['geometry'] = gdf_crown['geometry'].centroid

#set the centrod as the geometry
gdf_crown = gdf_crown.set_geometry('centroid')
#write the crown map to a file
gdf_crown.to_file(r"D:\PNM_crownmap_2025_centroid.shp")





import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

# Load the GeoDataFrame
path_crown = r"C:\Users\VasquezV\Downloads\panama_lianas_centroids_final\panama_lianas_centroids_final.shp"
gdf_crown = gpd.read_file(path_crown)

# Extract coordinates as a numpy array
coords = np.array(list(gdf_crown.geometry.apply(lambda geom: (geom.x, geom.y))))

# Build a cKDTree for fast spatial querying
tree = cKDTree(coords)

# Query the tree for the closest point (excluding itself)
distances, indices = tree.query(coords, k=2)  # k=2 includes self and nearest neighbor
closest_distances = distances[:, 1]  # Exclude the distance to itself (the first column)
closest_indices = indices[:, 1]

# Add results to the GeoDataFrame
gdf_crown['closest_distance'] = closest_distances
gdf_crown['closest_index'] = closest_indices

# Verify if points are evenly spaced by analyzing closest_distances
print(gdf_crown[['closest_distance', 'closest_index']].head())

# Save to file if needed
gdf_crown.to_file("output_with_distances.shp")

gdf_crown['notes'] = "Arturo Sanchez liana predicted"


crownmap_predicted= gpd.read_file(r"D:\PNM_crownmap_2025.shp")
crownmap_predicted= crownmap_predicted[['globalid', 'latin',
       'lianas','geometry']]

crownmap_predicted['geometry'] = crownmap_predicted['geometry'].centroid
import pandas as pd
gdf_crown.crs
crownmap_predicted.crs
crownmap_predicted= crownmap_predicted.to_crs(gdf_crown.crs)
combined_gdf= pd.concat([gdf_crown, crownmap_predicted], ignore_index=True)

coords = np.array(list(combined_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))

# Build a cKDTree for fast spatial querying
tree = cKDTree(coords)

# Query the tree for the closest point (excluding itself)
distances, indices = tree.query(coords, k=2)  # k=2 includes self and nearest neighbor
closest_distances = distances[:, 1]  # Exclude the distance to itself (the first column)
closest_indices = indices[:, 1]

# Add results to the GeoDataFrame
combined_gdf['closest_distance'] = closest_distances
combined_gdf['closest_index'] = closest_indices

# Verify if points are evenly spaced by analyzing closest_distances
print(combined_gdf[['closest_distance', 'closest_index']].head())

# Save to file if needed
combined_gdf.to_file("output_with_distances.shp")

final= gpd.read_file("output.shp")

final.columns
final= final[['Id', 'notes', 'globalid', 'latin', 'lianas', 'geometry']]
final.to_file(r"D:\PNM_centroids.shp")



reference_orig= r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\PNM_metrop_2024_12_20_M3E\cameras_position_fromMS_all.txt"
reference_orig= pd.read_csv(reference_orig, skiprows=1)
reference_orig      


reference_corrected= r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\PNM_metrop_2024_12_20_M3E\cameras_position_fromMS.txt"
reference_corrected= pd.read_csv(reference_corrected,skiprows=1)
reference_corrected

reference_merged= pd.merge(reference_orig, reference_corrected, on='#Label', how='left')
reference_merged['X/Longitude']= reference_merged['X/Longitude_y']
reference_merged['Y/Latitude']= reference_merged['Y/Latitude_y']
reference_merged['Z/Altitude']= reference_merged['Z/Altitude_y']

reference_merged=reference_merged[['#Label', 'Enable', 'X/Longitude', 'Y/Latitude',
       'Z/Altitude', 'Yaw', 'Pitch', 'Roll', 'Accuracy_X/Y/Z_(m)',
       'Accuracy_Yaw/Pitch/Roll_(deg)', 'Error_(m)', 'X_error', 'Y_error',
       'Z_error', 'Error_(deg)', 'Yaw_error', 'Pitch_error', 'Roll_error',
       'X_est', 'Y_est', 'Z_est', 'Yaw_est', 'Pitch_est', 'Roll_est', 'X_var',
       'Y_var', 'Z_var', 'Yaw_var', 'Pitch_var', 'Roll_var']]

reference_merged.to_csv(r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\PNM_metrop_2024_12_20_M3E\cameras_position_fromMS_all_corrected.txt", index=False)


crownmap_2025=r"D:\PNM_crownmap_2025.shp"
crownmap_2025= gpd.read_file(crownmap_2025)
crownmap_2025['latin'].unique()
crownmap_2025= crownmap_2025[['latin','flowering','tag', 'notes', 
       'lianas', 'illuminati', 'crown', 'dead_stand', 'new_leaves',
       'senecent_l', 'fruiting', 'leafing',   'geometry']]

crownmap_2025['about']= crownmap_2025['latin'] + " with tag " + crownmap_2025['tag']
crownmap_2025['aboutcrown']= "crown: "+ crownmap_2025['crown'] + " illumiantion: " + crownmap_2025['illuminati'] + " lianas: " + crownmap_2025['lianas'] 
crownmap_2025["aboutpheno"]= "flowering: " + crownmap_2025['flowering'] + ", new leaves: " + crownmap_2025['new_leaves'] + ", senecent leaves: " + crownmap_2025['senecent_l'] + ", fruiting: " + crownmap_2025['fruiting'] + ", leafing: " + str(crownmap_2025['leafing'])

crownmap_2025['dead_stand']
crownmap_2025= crownmap_2025[['about', 'aboutpheno', 'aboutcrown','notes','geometry']]
crownmap_2025["notes_old"]= crownmap_2025['notes']
crownmap_2025=crownmap_2025[['about', 'aboutpheno', 'aboutcrown','notes_old','geometry']]
crownmap_2025.to_file(r"D:\PNM_crownmap_2025_fieldmaps.shp")


from shapely.geometry import Polygon, LineString
sherman= r"D:\sherman2021.shp"
sherman= gpd.read_file(sherman)

def linestring_to_polygon(geom):
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return Polygon(coords)
    return geom

sherman['geometry'] = sherman['geometry'].apply(linestring_to_polygon)
sherman = sherman[sherman['geometry'].apply(lambda geom: isinstance(geom, (Polygon)))]
print(sherman)
sherman= sherman.to_crs('32617')
sherman.to_file(r"D:\sherman2021_poly.shp")

from shapely import box

bounds
sherman_shape= r"D:\sherman_shape.shp"
sherman_shape= gpd.read_file(sherman_shape)

sherman_shape.iloc[2].geometry.bounds
box1= box(sherman_shape.iloc[2].geometry.bounds[0], sherman_shape.iloc[2].geometry.bounds[1], sherman_shape.iloc[2].geometry.bounds[2], sherman_shape.iloc[2].geometry.bounds[3])

raster=r"D:\20250107_shermancrane_m3e_rgb.tif"
def crop_raster(input_path, output_path, shapely_polygon):
    with rasterio.open(input_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [shapely_polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

crop_raster(raster, r"D:\sherman2025_2.tif", box1)




from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
import rasterio
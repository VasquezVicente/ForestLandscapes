import os
import Metashape
import open3d as o3d
import geopandas as gpd
import pandas as pd
import laspy
import numpy as np
import re
bci_shape=r"D:/lidar_align/aux_files/50ha_shape.shp"
bci_shape=gpd.read_file(bci_shape)
print(bci_shape.bounds)

lidar_files=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\ALS"
#fine all laz starting with BCI
laz_files = [f for f in os.listdir(lidar_files) if f.endswith(".laz") and f.startswith("BCI")]
laz_table=pd.DataFrame(laz_files,columns=["laz_files"])
laz_table["xmin"]= laz_table["laz_files"].str.split(".").str[0].str.split("_").str[2].astype(float)
laz_table["ymin"]= laz_table["laz_files"].str.split(".").str[0].str.split("_").str[3].astype(float)

# i need tiles that intersect with xmin>= 625773.8594 and xmin<=626789.5653
# i need tiles that intersect with ymin>= 1.011743e+06 and ymin<=1.012276e+06

filtered_table= laz_table[(laz_table["xmin"]>=625000) & (laz_table["xmin"]<=626500) & (laz_table["ymin"]>=1011500) & (laz_table["ymin"]<=1012000)]

import shutil
for file in filtered_table["laz_files"]:
    print(file)
    shutil.copy(os.path.join(lidar_files,file),r"D:/lidar_align/aux_files/"+file)


# we need to combine the point clouds
list_tiles_main=os.listdir(r"D:/lidar_align/aux_files/")
path_tiles_main=r"D:/lidar_align/aux_files/"
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

xmin, ymin, xmax, ymax = bci_shape.bounds.values[0]
zmin=0
zmax=300
xmin, ymin, xmax, ymax= xmin-30, ymin-30, xmax+30, ymax+30
crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(xmin, ymin, zmin), max_bound=(xmax, ymax, zmax))

crop_lidar_cloud=combined_pcd.crop(crop_box)
o3d.visualization.draw_geometries([crop_lidar_cloud])
# Save the combined point cloud to a file (optional)
o3d.io.write_point_cloud("D:/lidar_align/aux_files/combined_dsm_point_cloud.ply", crop_lidar_cloud)


def icp(photo_cloud, lidar_cloud, threshold, identity_matrix,radius, max_nn, k):

    lidar_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    initial_eval_result = o3d.pipelines.registration.evaluate_registration(
        photo_cloud, lidar_cloud, threshold, identity_matrix)
    
    initial= str(initial_eval_result)
    initial_fitness_match = re.search(r'fitness=(.*?),', initial)
    initial_fitness_match = float(initial_fitness_match.group(1))
    final_fitness_match=initial_fitness_match+0.001
    while initial_fitness_match<1 and final_fitness_match>initial_fitness_match:
        #initial evaluation
        initial_eval_result = o3d.pipelines.registration.evaluate_registration(
            photo_cloud, lidar_cloud, threshold, identity_matrix)
        initial= str(initial_eval_result)
        initial_fitness_match = re.search(r'fitness=(.*?),', initial)
        initial_fitness_match = float(initial_fitness_match.group(1))
        #icp iteration
        loss = o3d.pipelines.registration.TukeyLoss(k=k)
        p2p_loss = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_result = o3d.pipelines.registration.registration_icp(
            photo_cloud, lidar_cloud, threshold, identity_matrix, p2p_loss)
        reg_result_str = str(reg_result)
        final_fitness_match = re.search(r'fitness=(.*?),', reg_result_str)
        final_fitness_match = float(final_fitness_match.group(1))
        if final_fitness_match<1 and final_fitness_match>initial_fitness_match:
            print('fitness improved from', initial_fitness_match, 'to', final_fitness_match)
            photo_cloud=photo_cloud.transform(reg_result.transformation)
            #save the reg result
            higuest_reg_result=reg_result
            higuest_reg_result_matrix=reg_result.transformation
            print('photo_cloud has been transformed')
        else:
            print('fitness did not improve, while loop will stop')
            print('final fitness is', final_fitness_match)
            print('the last photo cloud was not transformed,it is the correct one')
    return photo_cloud, lidar_cloud, higuest_reg_result, higuest_reg_result_matrix

threshold = 1
identity_matrix = np.identity(4)

photo_path=r"D:\lidar_align\data\BCI_50ha_2022_11_24_cloud.ply"
photo_cloud = o3d.io.read_point_cloud(photo_path)

photo_cloud1 = photo_cloud.crop(crop_box)

o3d.visualization.draw_geometries([crop_lidar_cloud])
import copy
def draw_registration_result(source, target, transformation): #its backwards
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])            #orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])        #blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

draw_registration_result(photo_cloud1, crop_lidar_cloud, identity_matrix)

photo_cloud_transformed,_,_,_ = icp(photo_cloud1, crop_lidar_cloud, threshold, identity_matrix, 4, 100, 3.7)
print("finish aligning the tile, saving it ")

#save all the clouds. 

# this function generates a dsm of the transformed cloud
#inputs the transformed cloud, and the path name of the new dsm
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


## we need to combine the point clouds lidar ==2. which is the DTM
# we combine here

#create surface_model_model(lidar==2cloud_dtm, dtm.tif)

# create the chm s using whatever code

# the result is that the chms are well aligned between each other


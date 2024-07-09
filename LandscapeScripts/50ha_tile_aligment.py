#basics 
import matplotlib.pyplot as plt
import os
import shutil
import rasterio
import numpy as np
import  cv2
from arosics import COREG, COREG_LOCAL
from LandscapeScripts.raster_tools import crop_raster
from LandscapeScripts.raster_tools import combine_ortho_dsm
import geopandas as gpd
from shapely.geometry import box as box1
import pandas as pd
import torch
from shapely.geometry import Polygon, MultiPolygon, box, shape
from shapely.ops import transform
import time
from rasterio.warp import reproject, Resampling
from datetime import datetime

wd_path= r"D:\BCI_50ha"
ortho_path = os.path.join(wd_path, "Orthophoto")
dsm_path = os.path.join(wd_path, "DSM")
product_path = os.path.join(wd_path, "Product")
cropped_path = os.path.join(wd_path, "Product_cropped")

BCI_50ha_shapefile = os.path.join(wd_path,"aux_files", "BCI_Plot_50ha.shp")
BCI_50ha = gpd.read_file(BCI_50ha_shapefile)
BCI_50ha.to_crs(epsg=32617, inplace=True)
BCI_50ha_buffer = box(BCI_50ha.bounds.minx-30, BCI_50ha.bounds.miny-30, BCI_50ha.bounds.maxx+30, BCI_50ha.bounds.maxy+30)  # Create a buffer around the plot

ortho_list = [file for file in os.listdir(ortho_path) if file.endswith(".tif")]
dsm_list= [file for file in os.listdir(dsm_path) if file.endswith(".tif")]  

#combine the orthomosaics and the dsm
for ortho, dsm in zip(ortho_list, dsm_list):
    ortho_file = os.path.join(ortho_path, ortho)
    dsm_file = os.path.join(dsm_path, dsm)
    out_file = os.path.join(wd_path, "Product", ortho)
    if not os.path.exists(out_file):
        combine_ortho_dsm(ortho_file, dsm_file, out_file)

#crop the orthomosaics
product_list = [file for file in os.listdir(product_path) if file.endswith(".tif")]
#crop the orthomosaic
for product in product_list:
    if not os.path.exists(os.path.join(cropped_path, product)):
        crop_raster(os.path.join(product_path, product), os.path.join(cropped_path, product), BCI_50ha_buffer)
#GLOBAL ALIGNMENT
reference1= os.path.join(cropped_path, ortho_list[69])
print("the referece is", reference1)
shutil.copy(reference1,os.path.join(wd_path,"Product_global", ortho_list[69]).replace("orthomosaic.tif","aligned_global.tif"))
successful_alignments = [file for file in os.listdir(os.path.join(wd_path, "Product_global")) if file.endswith(".tif")]
for orthomosaic in ortho_list[68::-1]:
    print(orthomosaic)
    if orthomosaic != ortho_list[69]:
        target = os.path.join(cropped_path, orthomosaic)
        global_path = target.replace("orthomosaic.tif","aligned_global.tif").replace("Product_cropped","Product_global")
        # Check if the file already exists in the Product_global directory
        if os.path.exists(global_path):
            print(f"Global alignment for {orthomosaic} already processed. Skipping...")
            continue
        kwargs2 = { 'path_out': global_path,
                    'fmt_out': 'GTIFF',
                    'r_b4match': 2,
                    's_b4match': 2,
                    'max_shift': 200,
                    'max_iter': 20,
                    'align_grids':True,
                    'match_gsd': True,
                    'binary_ws': False
                }
        alignment_successful = False
        while not alignment_successful and successful_alignments:
            try:
                CR= COREG(reference1, target, **kwargs2,ws=(2048,2048))
                CR.calculate_spatial_shifts()
                CR.correct_shifts()
                print("Global alignment successful")
                successful_alignments.append(global_path) # Add successful alignment to the list
                alignment_successful = True
            except:
                print("Global alignment failed, retrying with the previous successful alignment")
                reference1 = os.path.join(wd_path,"Product_global",successful_alignments.pop()) # Use the last successful alignment as reference

reference1= os.path.join(cropped_path, ortho_list[69])
successful_alignments = [reference1]
for orthomosaic in ortho_list[70:]:
    print(orthomosaic)
    if orthomosaic != ortho_list[69]:
        target = os.path.join(cropped_path, orthomosaic)
        global_path = target.replace("orthomosaic.tif","aligned_global.tif").replace("Product_cropped","Product_global")
        
        # Check if the file already exists in the Product_global directory
        if os.path.exists(global_path):
            print(f"Global alignment for {orthomosaic} already processed. Skipping...")
            continue

        kwargs2 = { 'path_out': global_path,
                'fmt_out': 'GTIFF',
                'r_b4match': 2,
                's_b4match': 2,
                'max_shift': 200,
                'max_iter': 20,
                'align_grids':True,
                'match_gsd': True,
                'binary_ws': False
            }
        alignment_successful = False
        while not alignment_successful and successful_alignments:
            try:
                CR= COREG(reference1, target, **kwargs2,ws=(2048,2048))
                CR.calculate_spatial_shifts()
                CR.correct_shifts()
                print("Global alignment successful")
                successful_alignments.append(global_path) # Add successful alignment to the list
                alignment_successful = True
            except:
                print("Global alignment failed, retrying with the previous successful alignment")
                reference1 = os.path.join(wd_path,"Product_global",successful_alignments.pop()) # Use the last successful alignment as reference

shutil.copy(os.path.join(cropped_path, ortho_list[69]),os.path.join(wd_path,"Product_global", ortho_list[69]).replace("orthomosaic.tif","aligned_global.tif"))


#align the global aligned versions, vertically
print("starting the vertical alignment")
os.makedirs(os.path.join(wd_path, 'Product_vertical'), exist_ok=True)
start_time = time.time() 
lidar_orthomosaic= os.path.join(wd_path,'aux_files','BCI_50ha_lidar_cropped_DTM_DEM.tif')
closest_date=r'BCI_50ha_2023_05_23_orthomosaic.tif'

with rasterio.open(os.path.join(wd_path,'Product_global',closest_date.replace('_orthomosaic.tif','_aligned_global.tif'))) as src:
    dem_data_photo = src.read(4)
    dem_meta_photo = src.meta

#resampling the lidar DEM to match the photogrammetry DEM
with rasterio.open(lidar_orthomosaic) as src:
    dem_meta_lidar = src.meta
    dem_resampled = np.zeros((dem_meta_lidar['count'], dem_meta_photo['height'], dem_meta_photo['width']), dtype=dem_meta_photo['dtype'])
    for band in range(dem_meta_lidar['count']):
        dem_data_lidar = src.read(band+1)
        reproject(
            dem_data_lidar, dem_resampled[band],
            src_transform=dem_meta_lidar['transform'],
            src_crs=dem_meta_lidar['crs'],
            dst_transform=dem_meta_photo['transform'],
            dst_crs=dem_meta_photo['crs'],
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0)
    dem_meta_lidar.update({'height': dem_meta_photo['height'],
                       'width': dem_meta_photo['width'],
                       'transform': dem_meta_photo['transform'],
                       'crs': dem_meta_photo['crs']})
    output_path3 = lidar_orthomosaic.replace("DTM_DEM.tif","DTM_DEM_resampled.tif")
    with rasterio.open(output_path3, 'w', **dem_meta_lidar) as dst:
        dst.write(dem_resampled)

print("finish resampling the lidar orthomosaic to match the photogrammetry orthomosaic")

#we read and get the np median without the nodata values  
with rasterio.open(lidar_orthomosaic.replace("DTM_DEM.tif","DTM_DEM_resampled.tif")) as src:
    dem_data_lidar = src.read(4)
    dem_meta_lidar = src.meta
    ref=np.median(dem_data_lidar[dem_data_lidar!=0])
with rasterio.open(os.path.join(wd_path,'Product_global',closest_date.replace('_orthomosaic.tif','_aligned_global.tif'))) as src:
    dem_data_photo = src.read(4)
    dem_meta_photo = src.meta
    tgt=np.median(dem_data_photo[dem_data_photo!=0])
    data=src.read()
    data[3,:,:]=data[3,:,:]+(ref-tgt)
    transform = src.transform
    meta = src.meta
    output_path3 = os.path.join(wd_path,'Product_vertical',closest_date.replace('_orthomosaic.tif','_aligned_global.tif'))
    with rasterio.open(output_path3, 'w', **meta) as dst:
        dst.write(data)

print("finish aligning vertically the closest date orthomosaic")

#list the horizontally aligened files

list_of_files = os.listdir(os.path.join(wd_path, 'Product_global'))
dates_files = [(datetime.strptime(f[9:19], '%Y_%m_%d'), f) for f in list_of_files if f.endswith('.tif')]
dates_files.sort()
sorted_files = [f for _, f in dates_files]

reference_main= os.path.join(wd_path,'Product_vertical',closest_date.replace('_orthomosaic.tif','_aligned_global.tif'))
#loop backwards for vertical aligment
for date in sorted_files[20::-1]:
    start_i=time.time()
    print("aligning vertically the date: ", date)
    with rasterio.open(reference_main) as src:
        dem_data_photo = src.read(4)
        dem_meta_photo = src.meta
        ref=np.median(dem_data_photo[dem_data_photo!=0])
        print("the median of the closest date is: ", ref)
    with rasterio.open(os.path.join(wd_path,'Product_global',date)) as src:
        dem_data_date = src.read()
        dem_meta_date = src.meta
        dem_resampled= np.zeros((dem_meta_date['count'], dem_meta_photo['height'], dem_meta_photo['width']), dtype=dem_meta_photo['dtype'])
        for band in range(dem_meta_date['count']):
            dem_data_date = src.read(band+1)
            reproject(
                dem_data_date, dem_resampled[band],
                src_transform=dem_meta_date['transform'],
                src_crs=dem_meta_date['crs'],
                dst_transform=dem_meta_photo['transform'],
                dst_crs=dem_meta_photo['crs'],
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0)
        dem_meta_date.update({'height': dem_meta_photo['height'],
                       'width': dem_meta_photo['width'],
                       'transform': dem_meta_photo['transform'],
                       'crs': dem_meta_photo['crs']})
        alpha_band = dem_resampled[-1]
        tgt = np.median(alpha_band[alpha_band != 0])
        print("the median of the date is: ", tgt)
        if ref> tgt:
            dem_resampled[3,:,:]=dem_resampled[3,:,:]+(ref-tgt)
        elif ref< tgt:
            dem_resampled[3,:,:]=dem_resampled[3,:,:]-(tgt-ref)
        output_path3 = os.path.join(wd_path,'Product_vertical',date)
        with rasterio.open(output_path3, 'w', **dem_meta_date) as dst:
            dst.write(dem_resampled)
        finish_i=time.time()
        reference_main=output_path3
        print("finish date in time: ", finish_i-start_i)
        print("finish backward aligment of date",date)
        
print("finish backward aligment")

def tile_ortho(sub, buffer, output_folder, gridInfo):
    with rasterio.open(sub) as src:
        for idx, row in gridInfo.iterrows():
            geom= row['geometry']
            geom2 = box(geom.bounds[0]-buffer, geom.bounds[1]-buffer, geom.bounds[2]+buffer, geom.bounds[3]+buffer)
            out_image, out_transform = rasterio.mask.mask(src, [geom2], crop=True)
            # Update metadata for the output raster
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            base_name = os.path.basename(sub)
            output_filename = f"{base_name.replace('aligned_global.tif', 'tile')}_{idx}.tif"
            filename=os.path.join(output_folder,output_filename)
            with rasterio.open(filename, "w", **out_meta) as dest:
                dest.write(out_image)

tile_folder_base = os.path.join(wd_path, "Product_tiles")
os.makedirs(tile_folder_base, exist_ok=True)
global_list= [file for file in os.listdir(os.path.join(wd_path, "Product_vertical")) if file.endswith(".tif")]

# Generate the grid once

grid= gpd.read_file(r"D:\BCI_50ha\aux_files\subplots_40by40.shp")

for num in range(0,len(global_list)):
    print(global_list[num])
    tile_folder1=os.path.join(tile_folder_base, f"{global_list[num].replace('_aligned_global.tif','')}")
    if not os.path.exists(tile_folder1):
        os.makedirs(tile_folder1)
    print(tile_folder1)
    tile_ortho(os.path.join(wd_path,"Product_vertical", global_list[num]),40,tile_folder1, grid)


folder_out= os.path.join(wd_path, "tiles_local")
os.makedirs(folder_out, exist_ok=True)

list_folder = [folder for folder in os.listdir(tile_folder_base) if os.path.isdir(os.path.join(tile_folder_base, folder))]

for index in range(0,100):
    print(f"Processing index {index}")
    #loop backwards
    reference= os.path.join(tile_folder_base, list_folder[21], f"{list_folder[21]}_tile_{index}.tif")
    for i in range(20, -1, -1):
        print(f"Processing {list_folder[i]}")
        target= os.path.join(tile_folder_base, list_folder[i], f"{list_folder[i]}_tile_{index}.tif")
        output_path = os.path.join(folder_out, list_folder[i])
        os.makedirs(output_path, exist_ok=True)
        output_path2 = os.path.join(output_path, f"{list_folder[i]}_tile_{index}.tif")
        kwargs = {
            'grid_res': 200,
            'window_size': (512, 512),
            'path_out': output_path2,
            'fmt_out': 'GTIFF',
            'q': False,
            'min_reliability': 30,
            'r_b4match': 2,
            's_b4match': 2,
            'max_shift': 100,
            'nodata': (0, 0),
            'match_gsd': True,
            'align_grids': True
        }
        try:
            if os.path.isfile(output_path2):
                print(f"File {output_path2} already exists")
                continue
            CR = COREG_LOCAL(reference, target, **kwargs)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            reference = output_path2
        except Exception as e:
            print(f"Error: {e}. Trying to align target to the previous reference.")
        
  
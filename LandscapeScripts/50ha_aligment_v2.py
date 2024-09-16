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
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge

wd_path= r"/home/vasquezv/BCI_50ha"
wd_path=r"D:\BCI_50ha"
ortho_path = os.path.join(wd_path, "Orthophoto")
dsm_path = os.path.join(wd_path, "DSM")
product_path = os.path.join(wd_path, "Product")
cropped_path = os.path.join(wd_path, "Product_cropped")
path_lidar= os.path.join(wd_path, "lidar")
path_aux= os.path.join(wd_path, "aux_files")
vertical_path= os.path.join(wd_path, "Vertical_local")
global_path= os.path.join(wd_path, "Vertical_global")

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
    else:
        print(f"Skipping {ortho} because it already exists")

#crop the orthomosaics
product_list = [file for file in os.listdir(product_path) if file.endswith(".tif")]
#crop the orthomosaic
for product in product_list:
    if not os.path.exists(os.path.join(cropped_path, product)):
        crop_raster(os.path.join(product_path, product), os.path.join(cropped_path, product), BCI_50ha_buffer)
    else:
        print(f"Skipping {product} because it already exists")

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

#LOCAL ALIGNMENT
from tqdm import tqdm

#LOCAL ALIGNMENT
global_path=os.path.join(wd_path,"Product_global")
ortho_list= [file for file in os.listdir(global_path) if file.endswith(".tif")]
ortho_list= sorted(ortho_list)
reference1= os.path.join(global_path, ortho_list[69])
print("the referece is", reference1)
local_path= os.path.join(wd_path,"Product_local")
shutil.copy(reference1,reference1.replace("Product_global","Product_local").replace("aligned_global.tif","aligned_local.tif"))

# Create a progress bar
pbar = tqdm(total=len(ortho_list), desc="Processing", ncols=100)

for orthomosaic in ortho_list[68::-1]:
    target= os.path.join(global_path, orthomosaic)
    print("the target is", target)
    out_path= target.replace("Product_global","Product_local").replace("aligned_global.tif","aligned_local.tif")
    if os.path.exists(out_path):
            print(f"Local alignment for {orthomosaic} already processed. Skipping...")
            pbar.update(1)
            continue
    kwargs = {          'grid_res': 200,
                        'window_size': (512, 512),
                        'path_out': out_path,
                        'fmt_out': 'GTIFF',
                        'q': False,
                        'min_reliability': 30,
                        'r_b4match': 2,
                        's_b4match': 2,
                        'max_shift': 100,
                        'nodata':(0, 0),
                        'match_gsd':False,
                    }
    CRL = COREG_LOCAL(reference1, target, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()
    CRL.CoRegPoints_table.to_csv(out_path.replace("aligned_local.tif","aligned.csv"))
    reference1= out_path
    pbar.update(1)

ortho_list= [file for file in os.listdir(global_path) if file.endswith(".tif")]
ortho_list= sorted(ortho_list)
reference1= os.path.join(global_path, ortho_list[69])

for orthomosaic in ortho_list[70:]:
    target= os.path.join(global_path, orthomosaic)
    out_path= target.replace("Product_global","Product_local").replace("aligned_global.tif","aligned_local.tif")
    if os.path.exists(out_path):
            print(f"Local alignment for {orthomosaic} already processed. Skipping...")
            pbar.update(1)
            continue
    kwargs = {          'grid_res': 200,
                        'window_size': (512, 512),
                        'path_out': out_path,
                        'fmt_out': 'GTIFF',
                        'q': False,
                        'min_reliability': 30,
                        'r_b4match': 2,
                        's_b4match': 2,
                        'max_shift': 100,
                        'nodata':(0, 0),
                        'match_gsd':False,
                    }
    CRL = COREG_LOCAL(reference1, target, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()
    CRL.CoRegPoints_table.to_csv(out_path.replace("aligned_local.tif","aligned.csv"))
    reference1= out_path
    pbar.update(1)

pbar.close()


#PERFORM A SECOND ALIGNMENT

#LOCAL ALIGNMENT
global_path=os.path.join(wd_path,"Product_local")
ortho_list= [file for file in os.listdir(global_path) if file.endswith(".tif")]
ortho_list= sorted(ortho_list)
reference1= os.path.join(global_path, ortho_list[69])
print("the referece is", reference1)
local_path= os.path.join(wd_path,"Product_local2")
os.makedirs(local_path, exist_ok=True)
shutil.copy(reference1,reference1.replace("Product_local","Product_local2").replace("aligned_local.tif","aligned_local2.tif"))

# Create a progress bar
pbar = tqdm(total=len(ortho_list), desc="Processing", ncols=100)

for orthomosaic in ortho_list[68::-1]:
    target= os.path.join(global_path, orthomosaic)
    print("the target is", target)
    out_path= target.replace("Product_local","Product_local2").replace("aligned_local.tif","aligned_local2.tif")
    if os.path.exists(out_path):
            print(f"Local alignment for {orthomosaic} already processed. Skipping...")
            pbar.update(1)
            continue
    kwargs = {          'grid_res': 200,
                        'window_size': (512, 512),
                        'path_out': out_path,
                        'fmt_out': 'GTIFF',
                        'q': False,
                        'min_reliability': 30,
                        'r_b4match': 2,
                        's_b4match': 2,
                        'max_shift': 100,
                        'nodata':(0, 0),
                        'match_gsd':False,
                    }
    CRL = COREG_LOCAL(reference1, target, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()
    CRL.CoRegPoints_table.to_csv(out_path.replace("aligned_local2.tif","aligned2.csv"))
    reference1= out_path
    pbar.update(1)

ortho_list= [file for file in os.listdir(global_path) if file.endswith(".tif")]
ortho_list= sorted(ortho_list)
reference1= os.path.join(global_path, ortho_list[69])

for orthomosaic in ortho_list[70:]:
    target= os.path.join(global_path, orthomosaic)
    out_path= target.replace("Product_local","Product_local2").replace("aligned_local.tif","aligned_local2.tif")
    if os.path.exists(out_path):
            print(f"Local alignment for {orthomosaic} already processed. Skipping...")
            pbar.update(1)
            continue
    kwargs = {          'grid_res': 200,
                        'window_size': (512, 512),
                        'path_out': out_path,
                        'fmt_out': 'GTIFF',
                        'q': False,
                        'min_reliability': 30,
                        'r_b4match': 2,
                        's_b4match': 2,
                        'max_shift': 100,
                        'nodata':(0, 0),
                        'match_gsd':False,
                    }
    CRL = COREG_LOCAL(reference1, target, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()
    CRL.CoRegPoints_table.to_csv(out_path.replace("aligned_local2.tif","aligned2.csv"))
    reference1= out_path
    pbar.update(1)

pbar.close()

#Perform vertical alignment of the global timeseries
#bring in the lidar data
list_of_tiles=os.listdir(path_lidar)
src_files_to_merge = []
for tile in list_of_tiles:
    tile_path = os.path.join(path_lidar, tile)
    src = rasterio.open(tile_path)
    src_files_to_merge.append(src)
mosaic, out_trans = merge(src_files_to_merge)
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
output_lidar_mosaic_50ha=os.path.join(path_aux, "BCI_50ha_lidar.tif")
with rasterio.open(output_lidar_mosaic_50ha, 'w', **out_meta) as dest:
    dest.write(mosaic)

#crop the lidar orthomosaic to the 50ha plot
with rasterio.open(output_lidar_mosaic_50ha) as src:
    out_image, out_transform = rasterio.mask.mask(src, [BCI_50ha_buffer], crop=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    output_lidar_mosaic_50ha_cropped = os.path.join(path_aux, "BCI_50ha_lidar_cropped.tif")
    with rasterio.open(output_lidar_mosaic_50ha_cropped, "w", **out_meta) as dest:
        dest.write(out_image)

#combine DEM and DTM
DTM=os.path.join(path_aux, "DTM_lidar_airborne.tif")
DEM=os.path.join(path_aux, "DEM_lidar_airborne_nonground.tif")
with rasterio.open(output_lidar_mosaic_50ha_cropped) as src:
    ortho_data_lidar = src.read()
    ortho_meta_lidar = src.meta.copy()
#crop the DTM to the 50ha plot
with rasterio.open(DTM) as src:
     out_image, out_transform = rasterio.mask.mask(src, [BCI_50ha_buffer], crop=True)
     out_meta = src.meta.copy()
     out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
     DTM_cropped= os.path.join(path_aux, "DTM_lidar_airborne_cropped.tif")
     with rasterio.open(DTM_cropped, "w", **out_meta) as dest:
        dest.write(out_image)
#reproject and deal with nodata values
with rasterio.open(DTM_cropped) as src:
    dtm_data_lidar = src.read(1)
    dtm_data_lidar= np.where(dtm_data_lidar==-32767, np.nan, dtm_data_lidar)
    dtm_meta = src.meta
resampled_dtm_lidar = np.zeros((ortho_meta_lidar['height'], ortho_meta_lidar['width']), dtype=dtm_data_lidar.dtype)#dtm_meta
reproject(
    dtm_data_lidar, resampled_dtm_lidar,
    src_transform=dtm_meta['transform'],
    src_crs=dtm_meta['crs'],
    dst_transform=ortho_meta_lidar['transform'],
    dst_crs=ortho_meta_lidar['crs'],
    resampling=Resampling.nearest)

#repeat previous steps with DEM
with rasterio.open(DEM) as src:
        out_image, out_transform = rasterio.mask.mask(src, [BCI_50ha_buffer], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        DEM_cropped= os.path.join(path_aux, "DEM_lidar_airborne_cropped.tif")
        with rasterio.open(DEM_cropped, "w", **out_meta) as dest:
            dest.write(out_image)
with rasterio.open(DEM) as src:
    dem_data_lidar = src.read(1)
    dem_meta = src.meta
resampled_dem_lidar = np.zeros((ortho_meta_lidar['height'], ortho_meta_lidar['width']), dtype=dem_data_lidar.dtype)
reproject(
    dem_data_lidar, resampled_dem_lidar,
    src_transform=dem_meta['transform'],
    src_crs=dem_meta['crs'],
    dst_transform=ortho_meta_lidar['transform'],
    dst_crs=ortho_meta_lidar['crs'],
    resampling=Resampling.nearest)

#combine the orthomosaic with the DTM and DEM
# Initialize a new array with an extra band
resampled_dem_lidar = resampled_dem_lidar[np.newaxis, :, :]
resampled_dtm_lidar = resampled_dtm_lidar[np.newaxis, :, :]
new_ortho_data_lidar = np.zeros((5, ortho_data_lidar.shape[1], ortho_data_lidar.shape[2]))
new_ortho_data_lidar[:3, :, :] = ortho_data_lidar[:3, :, :]
new_ortho_data_lidar[3, :, :] = resampled_dem_lidar[0, :, :]
new_ortho_data_lidar[4, :, :] = resampled_dtm_lidar[0, :, :]
print(new_ortho_data_lidar.shape)

ortho_meta_lidar.update(count=5) 
output_lidar_mosaic_50ha_cropped_DTM_DEM = os.path.join(path_aux, "BCI_50ha_lidar_cropped_DTM_DEM.tif")
with rasterio.open(output_lidar_mosaic_50ha_cropped_DTM_DEM, 'w', **ortho_meta_lidar) as dst:
    dst.write(new_ortho_data_lidar)


files_to_align= os.listdir(os.path.join(wd_path,"Product_global"))
lidar_orthomosaic= os.path.join(wd_path,'aux_files','output_lidar_mosaic_50ha_cropped_DTM_DEM.tif')

with rasterio.open(os.path.join(wd_path,"Product_global",files_to_align[69])) as src:
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

print("finish resampling the lidar orthomosaic")
#we read and get the np median without the nodata values  

with rasterio.open(lidar_orthomosaic.replace("DTM_DEM.tif","DTM_DEM_resampled.tif")) as src:
    dem_data_lidar = src.read(4)
    dem_meta_lidar = src.meta
    ref=np.median(dem_data_lidar[dem_data_lidar!=0])

with rasterio.open(os.path.join(wd_path,'Product_global',files_to_align[69])) as src:
    dem_data_photo = src.read(4)
    dem_meta_photo = src.meta
    tgt=np.median(dem_data_photo[dem_data_photo!=0])
    data=src.read()
    data[3,:,:]=data[3,:,:]+(ref-tgt)
    transform = src.transform
    meta = src.meta
    output_path3 = os.path.join(global_path,files_to_align[69].replace("aligned_global.tif","global.tif"))
    with rasterio.open(output_path3, 'w', **meta) as dst:
        dst.write(data)

print("finish aligning veritcally the closest date orthomosaic")


##global alignment
os.makedirs(os.path.join(wd_path,'Vertical_global'), exist_ok=True)
from datetime import datetime
list_of_files = os.listdir(os.path.join(wd_path, 'Product_global'))
dates_files = [(datetime.strptime(f[9:19], '%Y_%m_%d'), f) for f in list_of_files if f.endswith('.tif')]
dates_files.sort()
sorted_files = [f for _, f in dates_files]

print(sorted_files)
reference_main= os.path.join(wd_path,'Vertical_global',sorted_files[69].replace("aligned_global.tif","global.tif"))
for date in sorted_files[68::-1]:
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
        output_path3 = os.path.join(wd_path,'Vertical_global',date.replace("aligned_global.tif","global.tif"))
        with rasterio.open(output_path3, 'w', **dem_meta_date) as dst:
            dst.write(dem_resampled)
        reference_main=output_path3
        print("finish backward aligment of date",date)
        

print("finish foward aligment")
reference_main= os.path.join(wd_path,'Vertical_global',sorted_files[69].replace("aligned_global.tif","global.tif"))
for date in sorted_files[70:]:
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
        output_path3 = os.path.join(wd_path,'Vertical_global',date.replace("aligned_global.tif","global.tif"))
        with rasterio.open(output_path3, 'w', **dem_meta_date) as dst:
            dst.write(dem_resampled)
        print("finish date: ", date)
        reference_main=output_path3

#try both approaches, one for all to the main one and the other for the res
print("finish forward aligment")



##local vertical alignment
files_to_align= os.listdir(os.path.join(wd_path,"Product_local2"))
files_to_align= [file for file in files_to_align if file.endswith(".tif")]
print("finish housekeeping")
with rasterio.open(os.path.join(wd_path,"Product_local2",files_to_align[69])) as src:
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

print("finish resampling the lidar orthomosaic")
#we read and get the np median without the nodata values  

with rasterio.open(lidar_orthomosaic.replace("DTM_DEM.tif","DTM_DEM_resampled.tif")) as src:
    dem_data_lidar = src.read(4)
    dem_meta_lidar = src.meta
    ref=np.median(dem_data_lidar[dem_data_lidar!=0])

with rasterio.open(os.path.join(wd_path,'Product_local2',files_to_align[69])) as src:
    dem_data_photo = src.read(4)
    dem_meta_photo = src.meta
    tgt=np.median(dem_data_photo[dem_data_photo!=0])
    data=src.read()
    data[3,:,:]=data[3,:,:]+(ref-tgt)
    transform = src.transform
    meta = src.meta
    output_path3 = os.path.join(vertical_path,files_to_align[69].replace("aligned_local2.tif","local.tif"))
    with rasterio.open(output_path3, 'w', **meta) as dst:
        dst.write(data)

print("finish aligning vertically the closest date orthomosaic")

from datetime import datetime
os.makedirs(os.path.join(wd_path,'Vertical_local'), exist_ok=True)
list_of_files = os.listdir(os.path.join(wd_path, 'Product_local2'))
dates_files = [(datetime.strptime(f[9:19], '%Y_%m_%d'), f) for f in list_of_files if f.endswith('.tif')]
dates_files.sort()
sorted_files = [f for _, f in dates_files]

print(sorted_files)
reference_main= os.path.join(wd_path,'Vertical_local',sorted_files[69].replace("aligned_local2.tif","local.tif"))
for date in sorted_files[68::-1]:
    print("aligning vertically the date: ", date)
    with rasterio.open(reference_main) as src:
        dem_data_photo = src.read(4)
        dem_meta_photo = src.meta
        ref=np.median(dem_data_photo[dem_data_photo!=0])
        print("the median of the closest date is: ", ref)
    with rasterio.open(os.path.join(wd_path,'Product_local2',date)) as src:
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
        output_path3 = os.path.join(wd_path,'Vertical_global',date.replace("aligned_local2.tif","local.tif"))
        with rasterio.open(output_path3, 'w', **dem_meta_date) as dst:
            dst.write(dem_resampled)
        reference_main=output_path3
        print("finish backward aligment of date",date)
        

print("finish foward aligment")
reference_main= os.path.join(wd_path,'Vertical_local',sorted_files[69].replace("aligned_local2.tif","local.tif"))
for date in sorted_files[70:]:
    with rasterio.open(reference_main) as src:
        dem_data_photo = src.read(4)
        dem_meta_photo = src.meta
        ref=np.median(dem_data_photo[dem_data_photo!=0])
        print("the median of the closest date is: ", ref)
    with rasterio.open(os.path.join(wd_path,'Product_local2',date)) as src:
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
        output_path3 = os.path.join(wd_path,'Vertical_global',date.replace("aligned_local2.tif","local.tif"))
        with rasterio.open(output_path3, 'w', **dem_meta_date) as dst:
            dst.write(dem_resampled)
        print("finish date: ", date)
        reference_main=output_path3

#try both approaches, one for all to the main one and the other for the res
print("finish forward aligment")



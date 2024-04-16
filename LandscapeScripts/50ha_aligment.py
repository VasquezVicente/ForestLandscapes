#the following code is working without issues. pass all tests. 
import os
import time
import copy
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape, Polygon, MultiPolygon
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
from datetime import datetime
from rasterio.warp import calculate_default_transform, reproject, Resampling
from arosics import COREG, COREG_LOCAL
from shapely.geometry import box as box1
import matplotlib.pyplot as plt
import matplotlib	
matplotlib.use('TkAgg')

#Main directory
#wd_path = r"/home/vasquezv/BCI_50ha"
wd_path = r"D:\BCI_50ha"
#subdirectories
path_orthomosaic = os.path.join(wd_path, "Orthophoto")
path_DSM = os.path.join(wd_path, "DSM")
path_output = os.path.join(wd_path, "Product")
path_aux = os.path.join(wd_path, "aux_files")
path_lidar= os.path.join(wd_path, "lidar")
path_cropped= os.path.join(wd_path, "Product_cropped")
path_global= os.path.join(wd_path, "Product_global")
path_local= os.path.join(wd_path, "Product_local")
path_local_mean= os.path.join(wd_path, "Product_local_mean")
path_vertical= os.path.join(wd_path, "Product_vertical")

#create the directories
if not os.path.exists(path_output):
    os.makedirs(path_output)
if not os.path.exists(path_orthomosaic):   
    os.makedirs(path_orthomosaic)
if not os.path.exists(path_DSM):
    os.makedirs(path_DSM)
if not os.path.exists(path_aux):
    os.makedirs(path_aux)
if not os.path.exists(path_lidar):
    os.makedirs(path_lidar)
if not os.path.exists(path_global):
    os.makedirs(path_global)
if not os.path.exists(path_local):
    os.makedirs(path_local)
if not os.path.exists(path_local_mean):
    os.makedirs(path_local_mean)
if not os.path.exists(path_vertical):
    os.makedirs(path_vertical)

#read the 50ha shape file and transform it to UTM 17N
BCI_50ha_shapefile = os.path.join(path_aux, "BCI_Plot_50ha.shp")
BCI_50ha = gpd.read_file(BCI_50ha_shapefile)
BCI_50ha.to_crs(epsg=32617, inplace=True)
BCI_50ha_buffer = box(BCI_50ha.bounds.minx-20, BCI_50ha.bounds.miny-20, BCI_50ha.bounds.maxx+20, BCI_50ha.bounds.maxy+20)  # Create a buffer around the plot

#Create the lidar 2023 product for aligment
#provided on the data publication
#list tiles in lidar folder
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



#whole island combined dsm and orthomosaic function def



#combine the DSM and the orthomosaic
orthomosaics= os.listdir(path_orthomosaic)
DSMs= os.listdir(path_DSM)

DSMs_replaced = [filename.replace('dsm', 'orthomosaic') for filename in DSMs]
if DSMs_replaced == orthomosaics:
    print("The lists correspond to each other.")
else:
    print("The lists do not correspond to each other.")
for i in range(0, len(orthomosaics)):
    print(i)
    ortho= os.path.join(path_orthomosaic, orthomosaics[i])
    DSM= os.path.join(path_DSM, DSMs[i])
    with rasterio.open(ortho) as src:
        ortho_data = src.read()
        ortho_meta = src.meta.copy()
    with rasterio.open(DSM) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
    resampled_dem = np.zeros((ortho_meta['height'], ortho_meta['width']), dtype=dem_data.dtype)
    reproject(
        dem_data, resampled_dem,
        src_transform=dem_meta['transform'],
        src_crs=dem_meta['crs'],
        dst_transform=ortho_meta['transform'],
        dst_crs=ortho_meta['crs'],
        resampling=Resampling.nearest) 
    ortho_data[3, :, :] = resampled_dem
    ortho_meta.update(count=ortho_data.shape[0])
    out_file_name= os.path.join(path_output, orthomosaics[i])
    with rasterio.open(out_file_name, 'w', **ortho_meta) as dst:
            dst.write(ortho_data)
    print("finish combining the orthomosaics with the DSMs", i)

#crop all outputs to the shape of the 50ha plot
products= os.listdir(path_output)
for product in products:
    product_path= os.path.join(path_output, product)
    with rasterio.open(product_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [BCI_50ha_buffer], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        output_product= os.path.join(path_cropped, product)
        with rasterio.open(output_product, "w", **out_meta) as dest:
            dest.write(out_image)
    print("finish cropping the products", product)

print("finish cropping the products")
print("Starting the aligment process")
start_time = time.time() 

# Define the path to the working directory
wd_path = r"/home/vasquezv/BCI_50ha"
files_to_align= os.listdir(os.path.join(wd_path,'Product_cropped'))
lidar_orthomosaic= os.path.join(wd_path,'aux_files','BCI_50ha_lidar_cropped_DTM_DEM.tif')

print("the working directory is: ", wd_path)
print("the files to align are: ", files_to_align)
print("the lidar orthomosaic is: ", lidar_orthomosaic)

closest_date=r'BCI_50ha_2023_05_23_orthomosaic.tif'
closest_date_path=os.path.join(wd_path,'Product_cropped',closest_date)

target=os.path.join(wd_path,'Product_cropped',closest_date)
reference=lidar_orthomosaic

#locally correct the photogrammetry orthomosaic closer to the lidar
print("starting the local correction")
if not os.path.exists(os.path.join(wd_path,'Product_local')):
    os.makedirs(os.path.join(wd_path,'Product_local'))
output_path= os.path.join(wd_path,'Product_local',closest_date.replace('.tif','_local.tif'))
kwargs = {          'grid_res': 200,
                        'window_size': (512, 512),
                        'path_out': output_path,
                        'fmt_out': 'GTIFF',
                        'q': False,
                        'min_reliability': 30,
                        'r_b4match': 2,
                        's_b4match': 2,
                        'max_shift': 100,
                        'nodata':(0, 0),
                        'match_gsd':False
                    }
CRL = COREG_LOCAL(reference, target, **kwargs)
CRL.calculate_spatial_shifts()
CRL.correct_shifts()


#globally correct the photogrammetry orthomosaic with the closest date to the lidar
if not os.path.exists(os.path.join(wd_path,'Product_global')):
    os.makedirs(os.path.join(wd_path,'Product_global'))

CRL.CoRegPoints_table.to_csv(target.replace("orthomosaic.tif","local.csv").replace("Product_cropped","Product_local"))
points=CRL.CoRegPoints_table.sort_values('RELIABILITY', ascending=False).head(10)
xshift= points['X_SHIFT_M'].mean()
yshift= points['Y_SHIFT_M'].mean()

print("starting the global correction of the first orthomosaic")
for i in range(0,len(points)):
    nex=points['X_MAP'].iloc[i]
    newy=points['Y_MAP'].iloc[i]
     #ATTEMPT GLOBAL ALIGNMENT
    output_path2= target.replace("orthomosaic.tif","aligned_global.tif").replace("Product_cropped","Product_global")
    try:
            kwargs2 = {
                'path_out': output_path2,
                'fmt_out': 'GTIFF',
                'r_b4match': 2,
                's_b4match': 2,
                'max_shift': 100,
                'max_iter': 20,
                'align_grids':True,
                'match_gsd': False,
            }
            CR = COREG(reference, target,wp=(nex,newy),ws=(1024, 1024), **kwargs2)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            break  # Exit the loop if no RuntimeError
    except RuntimeError as e:
            print(f"Error processing {target}: {e}")
            continue  # Go to the next iteration if RuntimeError
print('finish the global alignment of first orthomosaic')


#Listing can be done regularly with os.lisdir(), however I needed to be absolutly sure that they were in the right order
list_of_files =  [f for f in os.listdir(os.path.join(wd_path,'Product_cropped')) if f.endswith('.tif')]
dates_files = [(datetime.strptime(f[9:19], '%Y_%m_%d'), f) for f in list_of_files if f.endswith('.tif')]
dates_files.sort()
list_dates = [f for _, f in dates_files]

reference=output_path2
reference2=output_path2

#backward loop starts in 2023_05_23 and 2018_04_04
for date in list_dates[69::-1]:
    print(date)
    target=os.path.join(wd_path,"Product_cropped", date)
    output_path= target.replace("orthomosaic.tif","local.tif").replace("Product_cropped","Product_local")
    kwargs = {          'grid_res': 200,
                        'window_size': (512, 512),
                        'path_out': output_path,
                        'fmt_out': 'GTIFF',
                        'q': False,
                        'min_reliability': 30,
                        'r_b4match': 2,  
                        's_b4match': 2,
                        'max_shift': 100,
                        'nodata':(0, 0),
                        'match_gsd':False,
                    }
    CRL = COREG_LOCAL(reference, target, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()

    CRL.CoRegPoints_table.to_csv(target.replace("orthomosaic.tif","aligned.csv").replace("Product_cropped","Product_local"))
    points=CRL.CoRegPoints_table.sort_values('RELIABILITY', ascending=False).head(10)
    for i in range(0,len(points)):
        nex=points['X_MAP'].iloc[i]
        newy=points['Y_MAP'].iloc[i]
        #ATTEMPT GLOBAL ALIGNMENT
        output_path2= target.replace("orthomosaic.tif","aligned_global.tif").replace("Product_cropped","Product_global")
        try:
            kwargs2 = {
                'path_out': output_path2,
                'fmt_out': 'GTIFF',
                'r_b4match': 2,
                's_b4match': 2,
                'max_shift': 100,
                'max_iter': 20,
                'align_grids':True,
                'match_gsd': False,
            }
            CR = COREG(reference, target,wp=(nex,newy),ws=(1024, 1024), **kwargs2)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            break  # Exit the loop if no RuntimeError
        except RuntimeError as e:
            print(f"Error processing {date}: {e}")
            continue  # Go to the next iteration if RuntimeError
    
    xshift= points['X_SHIFT_M'].mean()
    yshift= points['Y_SHIFT_M'].mean()

    #deshift the photogrammetry orthomosaic by applying the mean shift
    # Load the orthomosaic
    with rasterio.open(target, 'r+') as src:
        data=src.read()
        transform = src.transform
        new_transform = rasterio.Affine(transform.a, transform.b, transform.c + xshift,
                                        transform.d, transform.e, transform.f + yshift)
        meta = src.meta
        meta.update(transform=new_transform)

        output_path3 = target.replace("orthomosaic.tif","deshifted.tif").replace("Product_cropped","Product_global_mean")

        with rasterio.open(output_path3, 'w', **meta) as dst:
            dst.write(data)
    reference=output_path2
    print('finish the global alignment of the orthomosaic', date)

reference= reference2
#forward loop starts in 2023_05_23 and 2023_10_24
for date in list_dates[70:90]:
    print(date)
    target=os.path.join(wd_path,"Product_cropped", date)
    output_path= target.replace("orthomosaic.tif","local.tif").replace("Product_cropped","Product_local")
    kwargs = {          'grid_res': 200,
                        'window_size': (512, 512),
                        'path_out': output_path,
                        'fmt_out': 'GTIFF',
                        'q': False,
                        'min_reliability': 30,
                        'r_b4match': 2,
                        's_b4match': 2,
                        'max_shift': 100,
                        'nodata':(0, 0),
                        'match_gsd':False,
                    }
    CRL = COREG_LOCAL(reference, target, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()

    CRL.CoRegPoints_table.to_csv(target.replace("orthomosaic.tif","aligned.csv").replace("Product_cropped","Product_local"))
    points=CRL.CoRegPoints_table.sort_values('RELIABILITY', ascending=False).head(10)
    for i in range(0,len(points)):
        nex=points['X_MAP'].iloc[i]
        newy=points['Y_MAP'].iloc[i]
        #ATTEMPT GLOBAL ALIGNMENT
        output_path2= target.replace("orthomosaic.tif","aligned_global.tif").replace("Product_cropped","Product_global")
        try:
            kwargs2 = {
                'path_out': output_path2,
                'fmt_out': 'GTIFF',
                'r_b4match': 2,
                's_b4match': 2,
                'max_shift': 100,
                'max_iter': 20,
                'align_grids':True,
                'match_gsd': False,
            }
            CR = COREG(reference, target,wp=(nex,newy),ws=(1024, 1024), **kwargs2)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            break  # Exit the loop if no RuntimeError
        except RuntimeError as e:
            print(f"Error processing {date}: {e}")
            continue  # Go to the next iteration if RuntimeError  
    xshift= points['X_SHIFT_M'].mean()
    yshift= points['Y_SHIFT_M'].mean()
    #deshift the photogrammetry orthomosaic by applying the mean shift
    # Load the orthomosaic
    with rasterio.open(target, 'r+') as src:
        data=src.read()
        transform = src.transform
        new_transform = rasterio.Affine(transform.a, transform.b, transform.c + xshift,
                                        transform.d, transform.e, transform.f + yshift)
        meta = src.meta
        meta.update(transform=new_transform)
        output_path3 = target.replace("orthomosaic.tif","deshifted.tif").replace("Product_cropped","Product_global_mean")
        with rasterio.open(output_path3, 'w', **meta) as dst:
            dst.write(data)
    reference=output_path2
    print('finish the global alignment of the orthomosaic', date)

end_time = time.time()  # Stop the timer
elapsed_time = end_time - start_time  # Calculate the elapsed time
print("Time taken: {} seconds in aligment".format(elapsed_time))


#start the vertical aligment of the global products
print("starting the vertical alignment")
start_time = time.time() 
lidar_orthomosaic= os.path.join(wd_path,'aux_files','BCI_50ha_lidar_cropped_DTM_DEM.tif')
closest_date=r'BCI_50ha_2023_05_23_orthomosaic.tif'

#read the lidar DEM
print("finish housekeeping")
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
for date in sorted_files[69::-1]:
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
print("starting foward aligment")
reference_main= os.path.join(wd_path,'Product_vertical',closest_date.replace('_orthomosaic.tif','_aligned_global.tif'))
for date in sorted_files[70:90]:
    start_d=time.time()
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
        print("finish date: ", date)
        time_d=time.time()
        reference_main=output_path3
        print("finish in total time of date: ", start_d-time_d)

print("finish forward aligment")
finish_time = time.time()
print("the total time was: ", finish_time-start_time)


#vertical aligment of the local corrected orthomosaics
print("starting the vertical alignment")
start_time = time.time() 

# Define the path to the working directory
#erase the temporary orthomosaic from the first aligment local to lidar
path_to_erase=os.path.join(wd_path,'Product_local',closest_date.replace('.tif','_local.tif'))
os.remove(path_to_erase)

files_to_align= os.listdir(os.path.join(wd_path,'Product_cropped'))
lidar_orthomosaic= os.path.join(wd_path,'aux_files','output_lidar_mosaic_50ha_cropped_DTM_DEM.tif')
closest_date=r'BCI_50ha_2023_05_23_orthomosaic.tif'


#read the lidar DEM
print("finish housekeeping")
with rasterio.open(os.path.join(wd_path,'Product_local',closest_date.replace('_orthomosaic.tif','_local.tif'))) as src:
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
with rasterio.open(os.path.join(wd_path,'Product_local',closest_date.replace('_orthomosaic.tif','_local.tif'))) as src:
    dem_data_photo = src.read(4)
    dem_meta_photo = src.meta
    tgt=np.median(dem_data_photo[dem_data_photo!=0])
    data=src.read()
    data[3,:,:]=data[3,:,:]+(ref-tgt)
    transform = src.transform
    meta = src.meta
    output_path3 = os.path.join(wd_path,'Vertical_local',closest_date.replace('_orthomosaic.tif','_local.tif'))
    with rasterio.open(output_path3, 'w', **meta) as dst:
        dst.write(data)

print("finish aligning veritcally the closest date orthomosaic")
#list the horizontally aligened files
from datetime import datetime
list_of_files = os.listdir(os.path.join(wd_path, 'Product_local'))
dates_files = [(datetime.strptime(f[9:19], '%Y_%m_%d'), f) for f in list_of_files if f.endswith('.tif')]
dates_files.sort()
sorted_files = [f for _, f in dates_files]

print(sorted_files)
reference_main= os.path.join(wd_path,'Vertical_local',closest_date.replace('_orthomosaic.tif','_local.tif'))
for date in sorted_files[69::-1]:
    start_i=time.time()
    print("aligning vertically the date: ", date)
    with rasterio.open(reference_main) as src:
        dem_data_photo = src.read(4)
        dem_meta_photo = src.meta
        ref=np.median(dem_data_photo[dem_data_photo!=0])
        print("the median of the closest date is: ", ref)
    with rasterio.open(os.path.join(wd_path,'Product_local',date)) as src:
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
        output_path3 = os.path.join(wd_path,'Vertical_local',date)
        with rasterio.open(output_path3, 'w', **dem_meta_date) as dst:
            dst.write(dem_resampled)
        finish_i=time.time()
        reference_main=output_path3
        print("finish date in time: ", finish_i-start_i)
        print("finish backward aligment of date",date)
        

print("finish foward aligment")
print("starting foward aligment")
reference_main= os.path.join(wd_path,'Vertical_local',closest_date.replace('_orthomosaic.tif','_local.tif'))
for date in sorted_files[70:90]:
    start_d=time.time()
    with rasterio.open(reference_main) as src:
        dem_data_photo = src.read(4)
        dem_meta_photo = src.meta
        ref=np.median(dem_data_photo[dem_data_photo!=0])
        print("the median of the closest date is: ", ref)
    with rasterio.open(os.path.join(wd_path,'Product_local',date)) as src:
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
        output_path3 = os.path.join(wd_path,'Vertical_local',date)
        with rasterio.open(output_path3, 'w', **dem_meta_date) as dst:
            dst.write(dem_resampled)
        print("finish date: ", date)
        time_d=time.time()
        reference_main=output_path3
        print("finish in total time of date: ", start_d-time_d)
#try both approaches, one for all to the main one and the other for the rest

print("finish forward aligment")
finish_time = time.time()
print("the total time was: ", finish_time-start_time)
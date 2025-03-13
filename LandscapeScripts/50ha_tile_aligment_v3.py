#basics 
import os
import shutil
import rasterio
import numpy as np
import  cv2
from arosics import COREG, COREG_LOCAL
import geopandas as gpd
from shapely.geometry import box as box1
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, box, shape
from shapely.ops import transform
import time
from rasterio.warp import reproject, Resampling
from datetime import datetime
from rasterio.mask import mask

###FUNCTION
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
            output_filename = f"{base_name.replace('global.tif', 'tile')}_{idx}.tif"
            filename=os.path.join(output_folder,output_filename)
            with rasterio.open(filename, "w", **out_meta) as dest:
                dest.write(out_image)

wd_path= r"/home/vasquezv/BCI_50ha"

# Load the subplots
subplots= gpd.read_file(os.path.join(wd_path,"aux_files","subplots.shp"))

# Create a buffer around the plot
#lets use the global alignment products publish in vasquez et al 2024
global_orthomosaics= [file for file in os.listdir(os.path.join(wd_path, "orthomosaics_tile")) if file.endswith(".tif")]
print(f"Global orthomosaics: {global_orthomosaics}")

#Local alignemnt of the tiles
tiles_out= os.path.join(wd_path, "Product_tiles")
os.makedirs(tiles_out, exist_ok=True)

for orthomosaic in global_orthomosaics:
    name= orthomosaic.replace("_global.tif","")
    print(f"Processing {name}")
    os.makedirs(os.path.join(tiles_out,name), exist_ok=True)
    tile_ortho(os.path.join(wd_path,"orthomosaics_tile", orthomosaic),
               20,
               os.path.join(tiles_out,name),
               subplots)

#Local alignemnt of the tiles
local_tiles_out= os.path.join(wd_path, "Product_tiles_local")
os.makedirs(local_tiles_out, exist_ok=True)
list_folder = [folder for folder in os.listdir(tiles_out) if os.path.isdir(os.path.join(tiles_out, folder))]

#IMPORTANT
shutil.copytree(os.path.join(tiles_out, list_folder[0]), os.path.join(local_tiles_out, list_folder[0]))

#align the tiles horizontally
for index in range(0,50):
    print(f"Processing index {index}")
    #loop backwards
    reference= os.path.join(tiles_out, list_folder[0], f"{list_folder[0]}_tile_{index}.tif")
    for folder in range(1,32):
        print(f"Processing {list_folder[folder]}")
        target= os.path.join(tiles_out, list_folder[folder], f"{list_folder[folder]}_tile_{index}.tif")
        output_path = os.path.join(local_tiles_out, list_folder[folder])
        os.makedirs(output_path, exist_ok=True)
        output_path2 = os.path.join(output_path, f"{list_folder[folder]}_tile_{index}.tif")
        try:
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
            'align_grids': True}
            if os.path.isfile(output_path2):
                print(f"File {output_path2} already exists")
                continue
            CR = COREG_LOCAL(reference, target, **kwargs)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            reference = output_path2
        except Exception as e:
            print(f"Error: {e}")
        

#align the tiles vertically
#it doesnt matter which medium you choose since it is intrinsic alignment within this timeseries
#arguably if i choose the april date we would have a lower median since some trees are leafless 


local_tiles_vertical = os.path.join(wd_path, "Product_tiles_vertical")
os.makedirs(local_tiles_vertical, exist_ok=True)


list_folder = [folder for folder in os.listdir(local_tiles_out) if os.path.isdir(os.path.join(local_tiles_out, folder))]
shutil.copytree(os.path.join(local_tiles_out, list_folder[0]), os.path.join(local_tiles_vertical, list_folder[0]))


for index in range(0, 50):
    print(f"Processing index {index}")
    reference = os.path.join(tiles_out, list_folder[0], f"{list_folder[0]}_tile_{index}.tif")
    for folder in range(1, 32):
        print(f"Processing {list_folder[folder]}")
        target = os.path.join(local_tiles_out, list_folder[folder], f"{list_folder[folder]}_tile_{index}.tif")

        output_path = os.path.join(local_tiles_vertical, list_folder[folder])

        os.makedirs(output_path, exist_ok=True)

        output_path2 = os.path.join(output_path, f"{list_folder[folder]}_tile_{index}.tif")

        try:
            with rasterio.open(reference) as src:
                dem_data_photo = src.read(4)
                ref = np.median(dem_data_photo[dem_data_photo != 0]) ## why is not zer
                print("The median of the closest date is:", ref)
            
            with rasterio.open(target) as src:
                dem_data_date = src.read()
                dem_meta_date = src.meta
                alpha_band = dem_data_date[-1]
                tgt = np.median(alpha_band[alpha_band != 0])
                print("The median of the date is:", tgt)
                if ref> tgt:
                    dem_data_date[3,:,:]=dem_data_date[3,:,:]+(ref-tgt)
                elif ref< tgt:
                    dem_data_date[3,:,:]=dem_data_date[3,:,:]-(tgt-ref)
                                
                with rasterio.open(output_path2, 'w', **dem_meta_date) as dst:
                    dst.write(dem_data_date)
                
                reference = output_path2
                print("Finished backward alignment of date")
        except Exception as e:
            print(f"Error: {e}")
            continue

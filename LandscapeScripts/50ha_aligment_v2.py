#basics 
import matplotlib.pyplot as plt
import os
import shutil
import rasterio
import numpy as np
import  cv2
from arosics import COREG, COREG_LOCAL
from raster_tools import crop_raster
from raster_tools import combine_ortho_dsm
import geopandas as gpd
from shapely.geometry import box as box1
import pandas as pd
from segment_anything import SamPredictor, sam_model_registry
import torch
from shapely.geometry import Polygon, MultiPolygon, box, shape
from shapely.ops import transform

wd_path= r"/home/vasquezv/BCI_50ha"
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
reference1= os.path.join(global_path, ortho_list[69])
local_path= os.path.join(wd_path,"Product_local")
shutil.copy(reference1,reference1.replace("Product_global","Product_local").replace("aligned_global.tif","aligned_local.tif"))

# Create a progress bar
pbar = tqdm(total=len(ortho_list), desc="Processing", ncols=100)

for orthomosaic in ortho_list[68::-1]:
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

ortho_list= [file for file in os.listdir(global_path) if file.endswith(".tif")]
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
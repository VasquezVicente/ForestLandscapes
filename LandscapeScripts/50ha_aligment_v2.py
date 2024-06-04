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


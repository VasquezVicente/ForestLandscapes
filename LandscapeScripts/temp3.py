import os
import time
import copy
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from skimage.transform import rescale
from scipy.ndimage import zoom
from shapely.geometry import Polygon, MultiPolygon, box, shape
from matplotlib.patches import Rectangle
from segment_anything import SamPredictor, sam_model_registry
import torch
from rasterio.features import rasterize, geometry_mask, shapes
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
from rasterio import windows
from rasterio.plot import show
from rasterio.merge import merge
from datetime import datetime
from IPython.display import clear_output
from arosics import COREG, COREG_LOCAL
from shapely.geometry import box as box1
import os
from labelbox import Client
import labelbox
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
import json
import ndjson
import requests
import cv2
from typing import Dict, Any
import os
from labelbox import Client
import labelbox
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
import json
import ndjson
import requests
import cv2
from typing import Dict, Any
import matplotlib.animation as animation
BCI_50ha_directory = os.getcwd()
print(BCI_50ha_directory)

#BCI_50ha_directory = r"D:\BCI_50ha"
#functions
def combine_ortho_dsm(ortho_path,dsm_path, output_path):
    with rasterio.open(ortho_path) as src:
        ortho_data = src.read()
        ortho_meta = src.meta.copy()
    with rasterio.open(dsm_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
    resampled_dem = np.zeros((ortho_meta['height'], ortho_meta['width']), dtype=ortho_data.dtype)
    reproject(
    dem_data, resampled_dem,
    src_transform=dem_meta['transform'],
    src_crs=dem_meta['crs'],
    dst_transform=ortho_meta['transform'],
    dst_crs=ortho_meta['crs'],
    resampling=Resampling.nearest)
    ortho_data[3,:,:] = resampled_dem
    ortho_meta['count'] = 4
    with rasterio.open(output_path, 'w', **ortho_meta) as dst:
        dst.write(ortho_data)
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
def tile_ortho(sub, tile_size, buffer, output_folder):
    with rasterio.open(sub) as src:
        bounds = src.bounds
        xmin, ymin, xmax, ymax = bounds
        if tile_size <= 0:
            raise ValueError("tile_size must be greater than zero.")      
        x_range = xmax - xmin
        y_range = ymax - ymin
        x_tiles = int(np.ceil(x_range / tile_size))
        y_tiles = int(np.ceil(y_range / tile_size))
        x_residual = x_range % tile_size
        y_residual = y_range % tile_size
        if x_residual > 0:
            tile_size_x = tile_size + x_residual / x_tiles
        else:
            tile_size_x = tile_size
        if y_residual > 0:
            tile_size_y = tile_size + y_residual / y_tiles
        else:
            tile_size_y = tile_size
        if x_residual > 0 or y_residual > 0:
            print(f"Warning: Adjusted tile size used for residual coverage - X: {tile_size_x}, Y: {tile_size_y}")
        xmins = np.arange(xmin, (xmax - tile_size_x + 1), tile_size_x)
        xmaxs = np.arange((xmin + tile_size_x), xmax + 1, tile_size_x)
        ymins = np.arange(ymin, (ymax - tile_size_y + 1), tile_size_y)
        ymaxs = np.arange((ymin + tile_size_y), ymax + 1, tile_size_y)
        X, Y = np.meshgrid(xmins, ymins)
        Xmax, Ymax = np.meshgrid(xmaxs, ymaxs)
        gridInfo = pd.DataFrame({
            'xmin': X.flatten(),
            'ymin': Y.flatten(),
            'xmax': Xmax.flatten(),
            'ymax': Ymax.flatten(),
        })
        print(gridInfo)
    with rasterio.open(sub) as src:
        for idx, row in gridInfo.iterrows():
            geom2 = box1(row['xmin']-buffer, row['ymin']-buffer, row['xmax']+buffer, row['ymax']+buffer)
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
            output_filename = f"{base_name.replace('orthomosaic.tif', 'tile')}_{idx}.tif"
            filename=os.path.join(output_folder,output_filename)
            with rasterio.open(filename, "w", **out_meta) as dest:
                dest.write(out_image)
    return gridInfo


#50ha shapefile for boundaries
#read the 50ha shape file and transform it to UTM 17N
BCI_50ha_shapefile = os.path.join(BCI_50ha_directory,"aux_files", "BCI_Plot_50ha.shp")
BCI_50ha = gpd.read_file(BCI_50ha_shapefile)
BCI_50ha.to_crs(epsg=32617, inplace=True)
BCI_50ha_buffer = box(BCI_50ha.bounds.minx-20, BCI_50ha.bounds.miny-20, BCI_50ha.bounds.maxx+20, BCI_50ha.bounds.maxy+20)  # Create a buffer around the plot

#working directory

path_orthomosaic = os.path.join(BCI_50ha_directory, "Orthophoto")
path_DSM = os.path.join(BCI_50ha_directory, "DSM")
path_output= os.path.join(BCI_50ha_directory, "Product")
path_cropped= os.path.join(BCI_50ha_directory, "Product_cropped")
tile_folder_base= os.path.join(BCI_50ha_directory, "tiles")
base_output_path = os.path.join(BCI_50ha_directory, "output")
if not os.path.exists(path_orthomosaic):   
    os.makedirs(path_orthomosaic)
if not os.path.exists(path_DSM):
    os.makedirs(path_DSM)
if not os.path.exists(path_output):
    os.makedirs(path_output)
if not os.path.exists(path_cropped):
    os.makedirs(path_cropped)
if not os.path.exists(tile_folder_base):
    os.makedirs(tile_folder_base)
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)

#COMBINE DSM AND ORTHOPHOTO
orthomosaics = [filename for filename in os.listdir(path_orthomosaic) if filename.endswith('.tif')]
DSMs= [filename for filename in os.listdir(path_DSM) if filename.endswith('.tif')]
DSMs_replaced = [filename.replace('dsm', 'orthomosaic') for filename in DSMs]

if DSMs_replaced == orthomosaics:
    print("The lists correspond to each other.")
else:
    print("The lists do not correspond to each other.")

for i in range(0, len(orthomosaics)):
    ortho= os.path.join(path_orthomosaic, orthomosaics[i])
    DSM= os.path.join(path_DSM, DSMs[i])
    if not os.path.exists(os.path.join(path_cropped, orthomosaics[i])):
        combine_ortho_dsm(ortho, DSM, os.path.join(path_output, orthomosaics[i]))
        crop_raster(os.path.join(path_output, orthomosaics[i]), os.path.join(path_cropped, orthomosaics[i]), BCI_50ha_buffer)
        print(f"Combined {orthomosaics[i]} and {DSMs[i]}")
    else:
        print(f"File {orthomosaics[i]} already exists")



#tile the first 20 orthomosaics
orthomosaics= [filename for filename in os.listdir(path_cropped) if filename.endswith('.tif')]

for num in range(0,len(orthomosaics)):
    print(orthomosaics[num])
    tile_folder1=os.path.join(tile_folder_base, f"{orthomosaics[num].replace('.tif','')}")
    if not os.path.exists(tile_folder1):
        os.makedirs(tile_folder1)
    if len(os.listdir(tile_folder1)) == 50:
        print(f"Skipping {tile_folder1} because it already contains 49 files")
        continue
    print(os.path.join(path_cropped, orthomosaics[num]))
    print(tile_folder1)
    tile_ortho(os.path.join(path_cropped, orthomosaics[num]),100,20,tile_folder1)


# Define the list of folders
list_folder = [folder for folder in os.listdir(r"D:\BCI_50ha\tiles") if os.path.isdir(os.path.join(r"D:\BCI_50ha\tiles", folder))]
for index in range(0,50):
    print(f"Processing index {index}")
    til1 = None
    prev_til1 = None
    for i in range(1, len(list_folder)):
        if til1 is not None:
            prev_til1 = til1
        if til1 is None:
            til1 = os.path.join(r"D:\BCI_50ha\tiles", list_folder[i-1], f"{list_folder[i-1].replace('_orthomosaic','_tile')}_{index}.tif")
        else:
            til1 = os.path.join(base_output_path, list_folder[i-1],  f"{list_folder[i-1].replace('_orthomosaic','_tile')}_{index}.tif")
        til2 = os.path.join(r"D:\BCI_50ha\tiles", list_folder[i],  f"{list_folder[i].replace('_orthomosaic','_tile')}_{index}.tif")
        output_path = os.path.join(base_output_path, list_folder[i])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path2 = os.path.join(output_path, f"{list_folder[i].replace('_orthomosaic','_tile')}_{index}.tif")
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
            'match_gsd': False
        }
        try:
            if os.path.isfile(output_path2):
                print(f"File {output_path2} already exists")
                continue
            CRL = COREG_LOCAL(til1, til2, **kwargs)
            CRL.calculate_spatial_shifts()
            CRL.correct_shifts()
        except Exception as e:
            print(f"Error: {e}. Trying to align til2 to the previous til1.")
            if prev_til1 is not None:
                try:
                    if os.path.isfile(output_path2):
                        print(f"File {output_path2} already exists")
                        continue
                    CRL = COREG_LOCAL(prev_til1, til2, **kwargs)
                    CRL.calculate_spatial_shifts()
                    CRL.correct_shifts()
                except Exception as e:
                    print(f"Error: {e}. Failed to align til2 to the previous til1.")


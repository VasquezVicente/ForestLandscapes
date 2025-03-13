#basics 
import os
import shutil
import rasterio
import numpy as np
import  cv2
import arosics
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

def crop_raster(input_path, output_path, shapely_polygon):
    """
    Crops input raster to shape provided by a shapely polygon

    Parameters:
        Input_path= Full path to raster to be cropped(str) accepts os paths
        output_path= Full path of output file (str)
        shapely_polygon= accepts shapely polygon. it has to be iterable <Polygon>
    """
    with rasterio.open(input_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [shapely_polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

def align_orthomosaics(reference, target, target_out):
    """
    Aligns orthomosaics using arosics module, parameters are hard coded in this version. 

    Parameters:
        reference(str): Full path to orthomosaic to be used as reference orthomosaic
        target(str): Full path to target orthomosaic
        target_out(str): Full path of aligned orthomosaic
    """
    kwargs = {  'grid_res': 200,
                'window_size': (512, 512),
                'path_out': target_out,
                'fmt_out': 'GTIFF',
                'q': False,
                'min_reliability': 30,
                'r_b4match': 2,
                's_b4match': 2,
                'max_shift': 100,
                'nodata':(0, 0),
                'match_gsd':False,
                        }
    CRL = arosics.COREG_LOCAL(reference, target, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()
    print("finish align")



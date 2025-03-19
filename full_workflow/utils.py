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
def tile_ortho(sub, tile_size, buffer, output_folder):
    """
    Crops input raster to shape provided by a shapely polygon

    Parameters:
        sub= Full path to raster to be cropped(str) accepts os paths
        tile_size= tile size in meters (float)
        buffer= buffer in meters (float)
        output_folder= folder to save the tiles(path)  
    """
    #output folder should be changed to be a temporary file, to avoid issues with storage
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
            output_filename = f"output_raster_{idx}.tif"
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



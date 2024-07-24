#crown output
#crown segmentation
#utility modules
import matplotlib.pyplot as plt # plotting
import os # archivos del sistem
import pandas as pd #tablas
import rasterio # leer rasters
import cv2  # segemntacion
import numpy as np # matrices
import geopandas as gpd #geo dataframes
from shapely.geometry import Polygon, MultiPolygon #segmentacion
from shapely.geometry import box as box1 #segmentacion
from matplotlib.patches import Rectangle #plotting
import time # no es necesario
from shapely.geometry import Polygon, GeometryCollection #filtrar multipoligonos, y transformar a poligonos
from shapely.ops import transform # plotear

#rasterio functions
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio import windows
import uuid
import logging


wd_path= r"/home/vasquezv/BCI_50ha"
os.makedirs(os.path.join(wd_path,"segmented_crowns"), exist_ok=True)
all= gpd.read_file(os.path.join(wd_path,"BCI_50ha_crownmap_timeseries.shp"))
all=all.sort_values("date")

unique_identifiers= all["GlobalID"].unique()

for identifier in unique_identifiers:
    print(f"Processing crown {identifier}")
    crown_data = all[all["GlobalID"] == identifier]
    main_box = box1(crown_data.total_bounds[0] - 5, crown_data.total_bounds[1] - 5, crown_data.total_bounds[2] + 5, crown_data.total_bounds[3] + 5)

    latin_name = crown_data.iloc[0]["latin"]
    if not latin_name or isinstance(latin_name, float):
        crown_sp1 = "unknown"
        crown_sp2 = "unknown"
    else:
        latin_parts = latin_name.split(" ")
        crown_sp1 = latin_parts[0] if len(latin_parts) > 0 else "unknown"
        crown_sp2 = latin_parts[1] if len(latin_parts) > 1 else "unknown"
    
    tag = crown_data.iloc[0]["tag"]
    if tag == '000000':
        tag = crown_data.iloc[0]["GlobalID"]
    folder_out_crown = os.path.join(wd_path, "segmented_crowns", f"{crown_sp1}_{crown_sp2}_{tag}")
    os.makedirs(folder_out_crown, exist_ok=True)
    
    for date in crown_data["date"]:
        output_file_path = os.path.join(folder_out_crown, f"{date}.png")
        if os.path.exists(output_file_path):
            print(f"File {output_file_path} already exists. Skipping...")
            continue  # Skip the rest of the loop if file exists
        
        geom = crown_data[crown_data["date"] == date].iloc[0]["geometry"]
        with rasterio.open(os.path.join(wd_path, "Product_local2", f"BCI_50ha_{date}_aligned_local2.tif")) as src:
            out_image, out_transform = rasterio.mask.mask(src, [main_box], crop=True)
            out_meta = src.meta.copy()
            x_min, y_min = out_transform * (0, 0)
            xres, yres = out_transform[0], out_transform[4]
            if geom is None:
                print("Error: geom is None. Cannot proceed with transformation.")
            else:
                transformed_geom = transform(lambda x, y: ((x - x_min) / xres, (y - y_min) / yres), geom)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(out_image.transpose((1, 2, 0))[:, :, 0:3])
                ax.plot(*transformed_geom.exterior.xy, color='red')
                for interior in transformed_geom.interiors:
                    ax.plot(*interior.xy, color='red')
                ax.axis('off')
                fig.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)




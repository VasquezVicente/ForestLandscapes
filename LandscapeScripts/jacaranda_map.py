import geopandas as gpd
from scripts.raster_tools import calculate_purple_score
import os
import pandas as pd
import geopandas as gpd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import rasterio
import shapely
from shapely.geometry import box
from rasterio.mask import mask
import shapely.ops
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.merge import merge
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cv2
import numpy as np 
import math
from PIL import Image


def process_sample(sample, raster):
    unique_colors = set()
    for idx, row in sample.iterrows():
        geometry= row["geometry"]
        box_geom = box(*geometry.bounds)
        with rasterio.open(raster) as src:
            masked, out_transform = mask(src, [box_geom], crop=True)
            rgb_image = masked.transpose((1, 2, 0))[:, :, 0:3]
            flattened = rgb_image.reshape(-1, rgb_image.shape[-1])
            unique_colors.update(map(tuple, np.unique(flattened, axis=0)))
    return unique_colors


jacaranda_feb_2023 = r"D:\BCI_whole\Cropped\BCI_whole_2023_02_26_orthomosaic_jacaranda.tif"
jacaranda_mar_2023 = r"D:\BCI_whole\Cropped\BCI_whole_2023_03_18_orthomosaic_jacaranda.tif"
feb_2023_blue_sample=r"D:\BCI_whole\color_samples\jacaranda1_blue_sample.shp"
march_2023_blue_sample=r"D:\BCI_whole\color_samples\jacaranda2_blue_sample.shp"

blue_sample_feb_2023 = gpd.read_file(feb_2023_blue_sample)
blue_sample_mar_2023 = gpd.read_file(march_2023_blue_sample)

march_2023_colors = process_sample(blue_sample_mar_2023, jacaranda_mar_2023)
feb_2023_colors = process_sample(blue_sample_feb_2023, jacaranda_feb_2023)

jacaranda_unique_colors = march_2023_colors.union(feb_2023_colors)
jacaranda_unique_colors = {color for color in jacaranda_unique_colors if color != (0,0,0)}

size = int(math.ceil(math.sqrt(len(jacaranda_unique_colors))))
image = np.zeros((size-2, size, 3), dtype=np.uint8)
for i, color in enumerate(jacaranda_unique_colors):
    row = i // size
    col = i % size
    image[row, col] = color
plt.imshow(image)
plt.axis('off')
plt.show()


save_path = r"D:\BCI_whole\color_samples\jacaranda_unique_colors.tif"
with rasterio.open(save_path, 'w', driver='GTiff', width=image.shape[1], height=image.shape[0], count=3, dtype=image.dtype) as dst:
    dst.write(image.transpose(2, 0, 1))



#path to raw crown map
raw_feb=r"D:\BCI_whole\crowns_raw\BCI_whole_2023_02_26_orthomosaic_jacaranda.tif_crowns.shp"
raw_mar=r"D:\BCI_whole\crowns_raw\BCI_whole_2023_03_18_orthomosaic_jacaranda.tif_crowns.shp"
methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]

gdf_feb = gpd.read_file(raw_feb)
gdf_mar = gpd.read_file(raw_mar)

#load jacaranda sample
with rasterio.open(save_path) as src:
    jacaranda_data = src.read()
    jacaranda_data = jacaranda_data.transpose(1, 2, 0)
    plt.imshow(jacaranda_data)
    plt.axis('off')
    plt.show()

jacaranda_sample= cv2.cvtColor(jacaranda_data, cv2.COLOR_BGR2GRAY)
jacaranda_hist= cv2.calcHist([jacaranda_sample], [0], None, [256], [0, 256])
jacaranda_hist_normalize= cv2.normalize(jacaranda_hist, jacaranda_hist, 0, 1, cv2.NORM_MINMAX)

# Use jacaranda sample as the purple colors
purple_colors = [tuple(color) for color in jacaranda_data.reshape(-1, jacaranda_data.shape[-1]) if not np.all(color == 0)]
total_rows = len(gdf_feb)

for idx, (row_index, row) in enumerate(gdf_feb.iterrows()):
    geom= row["geometry"]
    with rasterio.open(jacaranda_feb_2023) as src:
        masked, out_transform = mask(src, [geom], crop=True)
        rgb_image = masked.transpose((1, 2, 0))[:, :, 0:3]
    jacaranda_instance= cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Create a mask for non-black pixels
    non_black_pixels_mask = (jacaranda_instance > 0).astype(np.uint8)
    total_pixels = np.prod(rgb_image.shape[:2])
    black_pixels = np.sum(np.all(rgb_image == [0, 0, 0], axis=-1))
    total_pixels = total_pixels - black_pixels

    purple_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    for color in purple_colors:
        color_mask = np.all(rgb_image == color, axis=-1)
        purple_mask = np.logical_or(purple_mask, color_mask)
    purple_pixels = np.sum(purple_mask)
    purple_score = (purple_pixels / total_pixels) * 100

    # Append the purple score to the DataFrame
    gdf_feb.loc[row_index, "purple_score"] = purple_score
    gdf_feb.loc[row_index, "purple_pixels"] = purple_pixels
    gdf_feb.loc[row_index, "total_pixels"] = total_pixels
    gdf_feb.loc[row_index, "black_pixels"] = black_pixels

    # Calculate histogram only for non-black pixels
    jacaranda_instance_hist= cv2.calcHist([jacaranda_instance], [0], non_black_pixels_mask, [256], [0, 256])
    jacaranda_instance_hist_normalize= cv2.normalize(jacaranda_instance_hist, jacaranda_instance_hist, 0, 1, cv2.NORM_MINMAX)
    for method in methods:
        similarity = cv2.compareHist(jacaranda_hist_normalize, jacaranda_instance_hist_normalize, method)
        gdf_feb.loc[row_index, f"similarity_{method}"] = similarity

    print(f"Progress: {idx+1}/{total_rows}")

gdf_feb.to_file(r"D:\BCI_whole\crowns_raw\BCI_whole_2023_02_26_crownmap_jacaranda.shp")


total_rows = len(gdf_mar)

for idx, (row_index, row) in enumerate(gdf_mar.iterrows()):
    geom= row["geometry"]
    with rasterio.open(jacaranda_feb_2023) as src:
        masked, out_transform = mask(src, [geom], crop=True)
        rgb_image = masked.transpose((1, 2, 0))[:, :, 0:3]
    jacaranda_instance= cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Create a mask for non-black pixels
    non_black_pixels_mask = (jacaranda_instance > 0).astype(np.uint8)
    total_pixels = np.prod(rgb_image.shape[:2])
    black_pixels = np.sum(np.all(rgb_image == [0, 0, 0], axis=-1))
    total_pixels = total_pixels - black_pixels

    purple_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    for color in purple_colors:
        color_mask = np.all(rgb_image == color, axis=-1)
        purple_mask = np.logical_or(purple_mask, color_mask)
    purple_pixels = np.sum(purple_mask)
    purple_score = (purple_pixels / total_pixels) * 100

    # Append the purple score to the DataFrame
    gdf_mar.loc[row_index, "purple_score"] = purple_score
    gdf_mar.loc[row_index, "purple_pixels"] = purple_pixels
    gdf_mar.loc[row_index, "total_pixels"] = total_pixels
    gdf_mar.loc[row_index, "black_pixels"] = black_pixels

    # Calculate histogram only for non-black pixels
    jacaranda_instance_hist= cv2.calcHist([jacaranda_instance], [0], non_black_pixels_mask, [256], [0, 256])
    jacaranda_instance_hist_normalize= cv2.normalize(jacaranda_instance_hist, jacaranda_instance_hist, 0, 1, cv2.NORM_MINMAX)
    for method in methods:
        similarity = cv2.compareHist(jacaranda_hist_normalize, jacaranda_instance_hist_normalize, method)
        gdf_mar.loc[row_index, f"similarity_{method}"] = similarity

    print(f"Progress: {idx+1}/{total_rows}")

gdf_mar.to_file(r"D:\BCI_whole\crowns_raw\BCI_whole_2023_03_18_crownmap_jacaranda.shp")
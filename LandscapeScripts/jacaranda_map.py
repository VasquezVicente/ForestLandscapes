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
image = np.zeros((size, size, 3), dtype=np.uint8)
for i, color in enumerate(jacaranda_unique_colors):
    row = i // size
    col = i % size
    image[row, col] = color
plt.imshow(image)
plt.axis('off')
plt.show()
save_path = r"D:\BCI_whole\color_samples\jacaranda_unique_colors.jpg"
cv2.imwrite(save_path, image)



#load samples
from rasterio.mask import mask
shps=r"D:\BCI_whole\color_samples\detectiontest3.shp"
samples = gpd.read_file(shps)
geom= samples["geometry"][4]
geom=box(*geom.bounds)
with rasterio.open(jacaranda_mar_2023) as src:
    out_image, out_transform = mask(src,[geom], crop=True)

plt.imshow(out_image.transpose((1, 2, 0))[:, :, 0:3])
plt.show()

import cv2
import numpy as np

# Load two images
image1 = cv2.imread(save_path)
image2 = cv2.imread("image2.jpg")

# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(out_image.transpose((1, 2, 0))[:, :, 0:3], cv2.COLOR_BGR2GRAY)

# Calculate histograms
hist1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

# Normalize histograms
hist1 = cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
hist2 = cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

# Compare histograms using different methods
methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
for method in methods:
    similarity = cv2.compareHist(hist1, hist2, method)
    print(f"Method: {method}, Similarity: {similarity}")

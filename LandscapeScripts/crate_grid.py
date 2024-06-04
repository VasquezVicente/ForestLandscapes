import os
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np

wd_path = r"D:\BCI_50ha"
BCI_50ha_shapefile = os.path.join(wd_path,"aux_files", "BCI_Plot_50ha.shp")
BCI_50ha = gpd.read_file(BCI_50ha_shapefile)
BCI_50ha.to_crs(epsg=32617, inplace=True)
BCI_50ha_buffer = box(BCI_50ha.bounds.minx-30, BCI_50ha.bounds.miny-30, BCI_50ha.bounds.maxx+30, BCI_50ha.bounds.maxy+30)  # Create a buffer around the plot


minx, miny, maxx, maxy = BCI_50ha_buffer.bounds
nx = int(np.ceil((maxx-minx) / 100))
ny = int(np.ceil((maxy-miny) / 100))

x = np.linspace(minx, maxx, nx+1)
y = np.linspace(miny, maxy, ny+1)
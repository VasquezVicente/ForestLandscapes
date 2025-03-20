import laspy
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

file=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\panama_BCI_plot2 0.010 m.las"
file_out=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\tiles"

xmin, xmax, ymin, ymax=-296.7343, 386.7407, -426.6668, 536.1783

def create_grid(xmin, xmax, ymin, ymax, tile_size, buffer=0):
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than zero.")
    
    x_range = xmax - xmin
    y_range = ymax - ymin
    x_tiles = int(np.ceil(x_range / tile_size))
    y_tiles = int(np.ceil(y_range / tile_size))

    x_residual = x_range % tile_size
    y_residual = y_range % tile_size

    tile_size_x = tile_size + x_residual / x_tiles if x_residual > 0 else tile_size
    tile_size_y = tile_size + y_residual / y_tiles if y_residual > 0 else tile_size

    if x_residual > 0 or y_residual > 0:
        print(f"Warning: Adjusted tile size used for residual coverage - X: {tile_size_x}, Y: {tile_size_y}")

    xmins = np.arange(xmin, xmax, tile_size_x)
    ymins = np.arange(ymin, ymax, tile_size_y)
    
    polygons = []
    for x in xmins:
        for y in ymins:
            poly = Polygon([
                (x - buffer, y - buffer), 
                (x + tile_size_x + buffer, y - buffer), 
                (x + tile_size_x + buffer, y + tile_size_y + buffer), 
                (x - buffer, y + tile_size_y + buffer)
            ])
            polygons.append(poly)

    # Create a GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")  # Change CRS if needed

    return grid_gdf
newgrid=create_grid(xmin, xmax, ymin, ymax, tile_size=100,buffer=0)

with laspy.open(file) as f:
    for i, row in newgrid.iterrows():
        subplot=[]
        for idx, points in enumerate(f.chunk_iterator(1000000), start=1):
            filter points falling inside the polygon
            move then into subplot
            subplot.append
        
        with laspy.open("grounds.laz", mode="w", header=f.header) as writer:
            for points in f.chunk_iterator(1_234_567):
                writer.write_points(points[points.classification == 2]

       


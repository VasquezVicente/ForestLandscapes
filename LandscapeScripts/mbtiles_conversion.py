import rasterio
import geopandas as gpd
import numpy as np
import rasterio.mask

path_shapefile=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Melvin\Panama_forest_plots.gpkg"
path=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Satellite\panama_elevation.tif"
path_output=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Melvin\SOBERANIA_p16_plot.tif"
shape=gpd.read_file(path_shapefile)
shape["Plot"].unique()
shape.columns
desired_plot= shape[shape['Plot']=='P16']

with rasterio.open(path) as src:
    out_image, out_transform = rasterio.mask.mask(src, desired_plot.geometry, crop=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    with rasterio.open(path_output, "w", **out_meta) as dest:
        dest.write(out_image)

import os
import rasterio
import geopandas as gpd
import rasterio.mask

#DJI pilot 2 does not support shapefiles or geopackages, so we need to convert the shapefile to a raster
#To this we make use of the Panama_forest_plots geopackage and the Panama elevation raster downloaded from ASTER NASA
#We will use the Sherman Administrative Polygon under STRI plot as an example

#First we read the geopackage using geopandas and filter the name of the plot we want to use
plots= gpd.read_file(r"\\stri-sm01\ForestLandscapes\UAVSHARE\Drone_Pilot_Data\Panama_forest_plots.gpkg")
plot=plots.loc[plots['Plot']=='Sherman Administrative Polygon under STRI']

#Then we read the elevation raster using rasterio and mask the raster using the plot geometry
elevation=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Drone_Pilot_Data\panama_elevation_resampled.tif"
with rasterio.open(elevation) as src:
    out_image, out_transform = rasterio.mask.mask(src, plot.geometry, crop=True)
    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open("sanlorenzo_elevation.tif", "w", **out_meta) as dest:
        dest.write(out_image)  

#commnad line GDAL translate to transform the raster to MBTiles
# gdal_translate -of MBTiles sanlorenzo_elevation.tif sanlorenzo_elevation.mbtiles


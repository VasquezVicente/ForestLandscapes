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

import geopandas as gpd
from shapely import box
import rasterio
from rasterio.transform import from_origin
import numpy as np
#DJI pilot 2 does not support shapefiles or geopackages, so we need to convert the shapefile to a raster
#To this we make use of the Panama_forest_plots geopackage and the Panama elevation raster downloaded from ASTER NASA
#We will use the Sherman Administrative Polygon under STRI plot as an example
from rasterio.crs import CRS

utm18s_wkt = """
PROJCS["WGS 84 / UTM zone 18S",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",-75],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",10000000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","32718"]]
"""


# Load and reproject the shapefile
file = r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\YFDP_Grid\line_start_end_XYToLine1.shp"
yasuni_grid = gpd.read_file(file)
yasuni_grid.to_crs(epsg=32718, inplace=True)
bounds = yasuni_grid.total_bounds  # minx, miny, maxx, maxy
yasuni_box = box(bounds[0]-40, bounds[1]-40, bounds[2]+40, bounds[3]+40)

# Create raster size based on 1 meter resolution
minx, miny, maxx, maxy = yasuni_box.bounds
width = int(np.ceil(maxx - minx))   
height = int(np.ceil(maxy - miny)) 

transform = from_origin(minx, maxy, 1, 1)  # (west, north, xsize, ysize)
raster_data = np.ones((height, width), dtype=np.uint8)
output_raster = r"D:/yasuni/mb_yasuni.tif"

# Write to GeoTIFF
with rasterio.open(
    output_raster,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster_data.dtype,
    crs = CRS.from_wkt(utm18s_wkt),
    transform=transform,
) as dst:
    dst.write(raster_data, 1)

print(f"Raster created: {output_raster}")

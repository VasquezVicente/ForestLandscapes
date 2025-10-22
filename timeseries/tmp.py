import rasterio
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.ops import unary_union

chm_path= r"\\stri-sm01\ForestLandscapes\LandscapeProducts\ALS\BCI_whole_2023_05_26_chm.tif"
ortho=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2023\BCI_whole_2023_06_19_EBEE_dipteryx\Orthophoto\BCI_whole_2023_06_19_orthomosaic_dipteryx.tif"

with rasterio.open(chm_path) as src:
    chm_data = src.read(1)
    plt.imshow(chm_data, cmap='viridis')
    plt.colorbar(label='Height (m)')
    plt.title('Canopy Height Model (CHM)')
    plt.show()
    # Create mask BEFORE filtering the data


from skimage import measure
from shapely.geometry import Polygon, unary_union

with rasterio.open(chm_path) as src:
    chm_data = src.read(1)
    transform = src.transform
    
    # Create binary mask
    land_mask = chm_data > 0
    
    # Find contours
    contours = measure.find_contours(land_mask, 0.5)
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            # Convert to real-world coordinates
            coords = []
            for point in contour:
                row, col = point
                x, y = rasterio.transform.xy(transform, row, col)
                coords.append((x, y))
            
            polygon = Polygon(coords)
            if polygon.is_valid:
                polygons.append(polygon)
    
    # Create final shoreline polygon
    #filter poygons of area less than 100 m2
    polygons = [poly for poly in polygons if poly.area >= 25]
    if polygons:
        shoreline_polygon = unary_union(polygons)

#plot the polygon

#put the polygons into a gdf
import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.GeoDataFrame(geometry=polygons)
gdf.crs = "EPSG:32617"  # Set the coordinate reference system
gdf.to_file("shoreline_polygons.shp")


gdf= gpd.read_file("shoreline_polygons.shp")

# Simplify to reduce vertices and smooth
tolerance = 3 # meters
shoreline_polygon = gdf.geometry[0].simplify(tolerance, preserve_topology=True)

fig, ax = plt.subplots(figsize=(10, 10))
gpd.GeoSeries(shoreline_polygon).plot(ax=ax, color='blue', alpha=0.5)
ax.set_title('Simplified Shoreline Polygon')
plt.show()


smooth_distance = 4 # meters
buffered_geom = gdf.geometry[0].buffer(-smooth_distance)

#choose the largest polygon
if buffered_geom.geom_type == 'MultiPolygon':
    # Get the largest polygon by area
    shoreline_polygon = max(buffered_geom.geoms, key=lambda x: x.area)
else:
    # It's already a single Polygon
    shoreline_polygon = buffered_geom

shoreline_polygon = shoreline_polygon.simplify(7, preserve_topology=True)

fig, ax = plt.subplots(figsize=(10, 10))
gpd.GeoSeries([shoreline_polygon]).plot(ax=ax, color='blue', alpha=0.5)
ax.set_title('Largest Smoothed Shoreline Polygon')
plt.show()


gpd2= gpd.GeoDataFrame(geometry=[shoreline_polygon])
gpd2.crs = "EPSG:32617"  # Set the coordinate reference
gpd2.to_file("shoreline_polygon.shp")
from rasterio.mask import mask 
import rasterio 
import numpy as np
with rasterio.open(ortho) as src:
    out_image, out_transform = mask(src, [shoreline_polygon], crop=True)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(np.transpose(out_image, (1, 2, 0)))
    ax.axis('off')  # Remove axes for cleaner image
    
    # Save 10x10 inches at 900 DPI
    plt.savefig('orthophoto_masked_2x2.png', dpi=900, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
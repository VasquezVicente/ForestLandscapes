import os
import geopandas as gpd
import json
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import pandas as pd

config_path = "crown-segment/config_combine.json"

with open(config_path, "r") as f:
        config = json.load(f)

#dummy data
polygons1 = [
    Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
    Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
]
crowns_date1 = gpd.GeoDataFrame({"geometry": polygons1}, crs="EPSG:32617")

# Create dummy data for crowns_date2
polygons2 = [
    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
    Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
    Polygon([(4, 4), (6, 4), (6, 6), (5, 6)])
]

crowns_date2 = gpd.GeoDataFrame({"geometry": polygons2}, crs="EPSG:32617")

#plt plot the polygons
fig, ax = plt.subplots()
crowns_date1.plot(ax=ax, color='red',alpha=0.5)
crowns_date2.plot(ax=ax, color='blue',alpha=0.5)
plt.show()
# Load the shapefile
crowns_date1=gpd.read_file(config["crowns_date1"])
crowns_date2=gpd.read_file(config["crowns_date2"])

#combine the two shapefiles using concat
combined_crowns=gpd.GeoDataFrame(pd.concat([crowns_date1,crowns_date2],ignore_index=True),crs="EPSG:32617")

intersections = gpd.overlay(crowns_date1, crowns_date2, how='intersection')


union_geometries = []

# Iterate through intersections and calculate IoU for each pair
for idx, intersection in intersections.iterrows():
    geom_A = crowns_date1.loc[crowns_date1.geometry.intersects(intersection.geometry), 'geometry'].values[0]
    geom_B = crowns_date2.loc[crowns_date2.geometry.intersects(intersection.geometry), 'geometry'].values[0]

    intersection_area = intersection.geometry.area
    union_area = geom_A.union(geom_B).area
    iou = intersection_area / union_area
    print(iou)

    if iou >= 0.5:
        union_geometries.append(geom_A.union(geom_B))

# Creating a new GeoDataFrame with the unioned geometries
union_gdf = gpd.GeoDataFrame(geometry=union_geometries, crs=crowns_date1.crs)

# If you want to see the resulting GeoDataFrame with unioned geometries
print(union_gdf)

fig, ax = plt.subplots()
#crowns_date1.plot(ax=ax, color='red',alpha=0.5)
#crowns_date2.plot(ax=ax, color='blue',alpha=0.5)
union_gdf.plot(ax=ax, color='green', alpha=0.5)
plt.show()
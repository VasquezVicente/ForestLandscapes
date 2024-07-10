#mbtiles for mavic
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import rasterio
import rasterio.mask
shapefile_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Melvin\Panama_forest_plots.gpkg"
plots=gpd.read_file(shapefile_path)

desired_plot="P16"
plot=plots[plots["Plot"]==desired_plot]

#I need a high resolution image of the plot

plt.figure(figsize=(10,10))
plt.imshow(out_image[0], cmap="terrain")
plt.show()









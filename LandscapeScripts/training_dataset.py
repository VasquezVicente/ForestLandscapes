#important urgent
import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.affinity import affine_transform
from shapely.ops import transform
from tqdm import tqdm

path_shps=r"D:\BCI_50ha\crown_segmentation"
shps=[os.path.join(path_shps,shp) for shp in os.listdir(path_shps) if shp.endswith("_improved.shp")]
#list all the locally aligned orthomosaics
paths_ortho=r"D:\BCI_50ha\timeseries_local_alignment"
orthos=[os.path.join(paths_ortho,ortho) for ortho in os.listdir(paths_ortho) if ortho.endswith(".tif")]

#we can combine all the shapefiles into one
all_shapefiles=gpd.GeoDataFrame()
for shapefile in shps:
    shapefile_subset=gpd.read_file(shapefile)
    shapefile_subset=shapefile_subset[["tag","area","score","geometry","Mnemonic","Latin"]]
    #add the date as a column
    shapefile_subset["date"]="_".join(shapefile.split("\\")[-1].split("_")[0:3])
    all_shapefiles=pd.concat([all_shapefiles,shapefile_subset])
    print("finished",shapefile)


if not os.path.exists(r"D:\BCI_50ha\crown_segmentation\crown_images_with_overlay"):
    os.makedirs(r"D:\BCI_50ha\crown_segmentation\crown_images_with_overlay")

if not os.path.exists(r"D:\BCI_50ha\crown_segmentation\crown_images_no_overlay"):
    os.makedirs(r"D:\BCI_50ha\crown_segmentation\crown_images_no_overlay")

# Iterate over each species

all_shapefiles=all_shapefiles[all_shapefiles["tag"]!=-9999]
all_shapefiles=  all_shapefiles[all_shapefiles["Latin"].notna()]

#get all tags
tags=all_shapefiles["tag"].unique()
species=all_shapefiles["Latin"].unique()

for sp in species:
    if not os.path.exists(os.path.join(r"D:\BCI_50ha\crown_segmentation\crown_images_with_overlay", f"{sp}")):
        os.makedirs(os.path.join(r"D:\BCI_50ha\crown_segmentation\crown_images_with_overlay", f"{sp}"))
    if not os.path.exists(os.path.join(r"D:\BCI_50ha\crown_segmentation\crown_images_no_overlay", f"{sp}")):
        os.makedirs(os.path.join(r"D:\BCI_50ha\crown_segmentation\crown_images_no_overlay", f"{sp}"))



# i need to check if the file exist then process if not then writte the file
for sp in species:
    species_tag = all_shapefiles[all_shapefiles["Latin"] == sp]["tag"].unique()
    for tag_crown in species_tag:
        print(tag_crown)
        crown=all_shapefiles[all_shapefiles["tag"]==tag_crown]
        crown_sp=crown["Latin"].unique()[0]
        for index,instance in crown.iterrows():
            print(instance)
            geom_box = instance["geometry"].bounds
            geom_box = box(geom_box[0]-5, geom_box[1]-5, geom_box[2]+5, geom_box[3]+5)
            corresponding_ortho=[ortho for ortho in orthos if instance["date"] in ortho]
            print(corresponding_ortho) 
            with rasterio.open(corresponding_ortho[0]) as src:
                out_image, out_transform = mask(src, [geom_box], crop=True)
                x_min, y_min = out_transform * (0, 0)
                xres, yres = out_transform[0], out_transform[4]
                if isinstance(instance["geometry"], MultiPolygon):
                    continue
                transformed_geom = transform(lambda x, y: ((x-x_min)/xres, (y-y_min)/yres), instance["geometry"])
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(out_image.transpose((1, 2, 0))[:,:,0:3])
                ax.plot(*transformed_geom.exterior.xy, color='red')
                ax.axis('off')
                outpath_no_overlay = os.path.join(r"D:\BCI_50ha\crown_segmentation\crown_images_no_overlay",f"{crown_sp}", f"{instance['tag']}_{instance['date']}.png")
                fig.savefig(outpath_no_overlay, bbox_inches='tight', pad_inches=0)
                plt.close(fig)  # Close the figure to avoid memory leaks



    

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

#omit the two species that were already sent
all_shapefiles=all_shapefiles[all_shapefiles["Latin"]!="Alchornea costaricensis"]
all_shapefiles=all_shapefiles[all_shapefiles["Latin"]!="Prioria copaifera"]

all_shapefiles["area"]
plt.hist(all_shapefiles["area"],bins=100)
plt.show()

#get all crowns above 500 m2
tags_500=all_shapefiles[all_shapefiles["area"]>500]["tag"].unique()
print(tags_500)


if not os.path.exists(r"D:\BCI_50ha\crown_segmentation\crown_images_500"):
    os.makedirs(r"D:\BCI_50ha\crown_segmentation\crown_images_500")

if not os.path.exists(r"D:\BCI_50ha\crown_segmentation\crown_images_with_overlay"):
    os.makedirs(r"D:\BCI_50ha\crown_segmentation\crown_images_with_overlay")

if not os.path.exists(r"D:\BCI_50ha\crown_segmentation\crown_images_no_overlay"):
    os.makedirs(r"D:\BCI_50ha\crown_segmentation\crown_images_no_overlay")

# Iterate over each species

for tag in tags_500:
    print(tag)
    species_tag = all_shapefiles[all_shapefiles["tag"] == tag]
    #skip if species tag Latin is NaN
    if species_tag["Latin"].isna().any():
        continue
    for index,instance in species_tag.iterrows():
            crown_sp=instance["Latin"]
            crown_sp1= crown_sp.split(" ")[0]
            crown_sp2= crown_sp.split(" ")[1]
            outpath_no_overlay = os.path.join(r"D:\BCI_50ha\crown_segmentation\crown_images_500",f"{crown_sp1}_{crown_sp2}_{instance['tag']}_{instance['date']}.png")
            if not os.path.exists(outpath_no_overlay):
                print(outpath_no_overlay)
                geom_box = instance["geometry"].bounds
                geom_box = box(geom_box[0]-5, geom_box[1]-5, geom_box[2]+5, geom_box[3]+5)
                corresponding_ortho=[ortho for ortho in orthos if instance["date"] in ortho]
                print(corresponding_ortho) 
                with rasterio.open(corresponding_ortho[0]) as src:
                    out_image, out_transform = mask(src, [geom_box], crop=True)
                    x_min, y_min = out_transform * (0, 0)
                    xres, yres = out_transform[0], out_transform[4]
                    transformed_geom = transform(lambda x, y: ((x-x_min)/xres, (y-y_min)/yres), instance["geometry"])
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(out_image.transpose((1, 2, 0))[:,:,0:3])
                    ax.plot(*transformed_geom.exterior.xy, color='red')
                    for interior in transformed_geom.interiors:
                        ax.plot(*interior.xy, color='red')
                    ax.axis('off')
                    fig.savefig(outpath_no_overlay, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
            else:
                print("file already exists")


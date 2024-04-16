import rasterio
from rasterio.mask import mask
import os
import shapely
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patches as patches
from shapely.affinity import affine_transform
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np


#list all the segmented shps
path_shps=r"D:\BCI_50ha\crown_segmentation"
reference_raw=r"D:\BCI_50ha\crown_segmentation\BCI_50ha_2020_08_01_crownmap_raw\Crowns_2020_08_01_MergedWithPlotData.shp"
shps=[os.path.join(path_shps,shp) for shp in os.listdir(path_shps) if shp.endswith("_improved.shp")]
#list all the locally aligned orthomosaics
paths_ortho=r"D:\BCI_50ha\timeseries_local_alignment"
orthos=[os.path.join(paths_ortho,ortho) for ortho in os.listdir(paths_ortho) if ortho.endswith(".tif")]

#we need to know the level of noise of the segmentation

#we can combine all the shapefiles into one
all_shapefiles=gpd.GeoDataFrame()
for shapefile in shps:
    shapefile_subset=gpd.read_file(shapefile)
    shapefile_subset=shapefile_subset[["tag","area","score","geometry","Mnemonic","Latin"]]
    #add the date as a column
    shapefile_subset["date"]="_".join(shapefile.split("\\")[-1].split("_")[0:3])
    all_shapefiles=all_shapefiles.append(shapefile_subset)
    print("finished",shapefile)

all_shapefiles["Latin"].unique()   


tag_k=106463
subset_k=all_shapefiles[all_shapefiles["tag"]==tag_k]
#plot the area against the date
plt.plot(subset_k["date"],subset_k["area"])
plt.xticks(rotation=90)
plt.show()
print(subset_k["tag"].unique())



referece_shape= gpd.read_file(reference_raw)
referece_shape=referece_shape.to_crs("EPSG:32617")
crown=referece_shape[referece_shape["Tag"]==str(tag_k)]
box_bounds=crown.geometry.total_bounds
box_bounds=box(box_bounds[0]-5, box_bounds[1]-5, box_bounds[2]+5, box_bounds[3]+5)

for i in range(0, len(orthos),10):
    with rasterio.open(orthos[i]) as src:
        masked_image, out_transform = mask(src,[box_bounds], crop=True)
        x_min, y_min = out_transform * (0, 0)
        xres, yres = out_transform[0], out_transform[4]
        transformed_geom = crown.geometry.apply(lambda geom: shapely.ops.transform(lambda x, y: ((x-x_min)/xres, (y-y_min)/yres), geom))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(masked_image.transpose((1, 2, 0))[:,:,0:3])
        transformed_geom.boundary.plot(ax=ax, edgecolor='red', linewidth=2)
        plt.show()


#lets do a summary of the data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd
import shapely.ops
from shapely.geometry import box
import rasterio
from rasterio.mask import mask

def transform_geometry(geom, transform):
    x_min, y_min = transform * (0, 0)
    xres, yres = transform[0], transform[4]
    return shapely.ops.transform(lambda x, y: ((x - x_min) / xres, (y - y_min) / yres), geom)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
artists = []
transformed_geom = crown.geometry.apply(transform_geometry, transform=out_transform)

for i in range(0, len(orthos)):
    with rasterio.open(orthos[i]) as src:
        masked_image, _ = mask(src, [box_bounds], crop=True)
        image = ax.imshow(masked_image.transpose((1, 2, 0))[:, :, 0:3])
        transformed_geom.boundary.plot(ax=ax, edgecolor='red', linewidth=2)
        artists.append([image, ax.collections[-1]])  # Use ax.collections[-1] to get the last plotted polygon artist
# Create animation
ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)
ani.save('animation.mp4', writer='ffmpeg_file')
plt.show()


# we need the number of crown per species that hit 500m2 and more

tags_above_500 = all_shapefiles[all_shapefiles["area"] > 500]["tag"].unique()
subset_above_500 = all_shapefiles[all_shapefiles["tag"].isin(tags_above_500)]
per_species_500 = subset_above_500.groupby('Latin')['tag'].nunique()
print("Tags per species for area > 500 m2:")
print(per_species_500)

tags_above_200 = all_shapefiles[all_shapefiles["area"] > 200]["tag"].unique()
subset_above_200 = all_shapefiles[all_shapefiles["tag"].isin(tags_above_200)]
per_species_200 = subset_above_200.groupby('Latin')['tag'].nunique()
print("\nTags per species for area > 200 m2:")
print(per_species_200)

tags_above_100 = all_shapefiles[all_shapefiles["area"] > 100]["tag"].unique()
subset_above_100 = all_shapefiles[all_shapefiles["tag"].isin(tags_above_100)]
per_species_100 = subset_above_100.groupby('Latin')['tag'].nunique()
print("\nTags per species for area > 100 m2:")
print(per_species_100)

summary_df = pd.DataFrame(index=all_shapefiles['Latin'].unique())

# Add the results as new columns to the DataFrame
summary_df['Tags > 500 m2'] = per_species_500
summary_df['Tags > 200 m2'] = per_species_200
summary_df['Tags > 100 m2'] = per_species_100

# Fill NaN values with 0
summary_df = summary_df.fillna(0)

# Convert the counts to integers
summary_df = summary_df.astype(int)

print(summary_df)
summary_df.to_csv("summary.csv")

# i need the most abundant species
top_10_species=all_shapefiles["Latin"].value_counts().index[0:10]
print(top_10_species)

quararibeas=all_shapefiles[all_shapefiles["Latin"]=="Quararibea asterolepis"]
orthos=[os.path.join(paths_ortho,ortho) for ortho in os.listdir(paths_ortho) if ortho.endswith(".tif")]

all_arrays=[]
for index, tree in quararibeas.iterrows():
    print(tree["date"])
    correspondent_ortho= [ortho for ortho in orthos if tree["date"] in ortho]
    print(correspondent_ortho)
    with rasterio.open(correspondent_ortho[0]) as src:
        out_image, out_transform = mask(src,[tree["geometry"]], crop=True)
        mask_nonzero = out_image != 0
        all_arrays.append(out_image)

red_means=[]
green_means=[]
blue_means=[]
for tree in all_arrays:
    flatten_tree_red=tree[0,:,:].flatten()
    flatten_tree_green=tree[1,:,:].flatten()
    flatten_tree_blue=tree[2,:,:].flatten()
    red_means.append(flatten_tree_red[flatten_tree_red!=0].mean())
    green_means.append(flatten_tree_green[flatten_tree_green!=0].mean())
    blue_means.append(flatten_tree_blue[flatten_tree_blue!=0].mean())


plt.hist(red_means, bins=50, color='r', alpha=0.5)
plt.hist(green_means, bins=50, color='g', alpha=0.5)
plt.hist(blue_means, bins=50, color='b', alpha=0.5)
plt.show()


#lets do the same for the 10 most abundant species
alseis=all_shapefiles[all_shapefiles["Latin"]=="Alseis blackiana"]

all_arrays_alseis=[]
for index, tree in alseis.iterrows():
    print(tree["date"])
    geom=tree["geometry"]
    if geom is None:
        continue
    correspondent_ortho= [ortho for ortho in orthos if tree["date"] in ortho]
    print(correspondent_ortho)
    with rasterio.open(correspondent_ortho[0]) as src:
        out_image, out_transform = mask(src,[geom], crop=True)
        mask_nonzero = out_image != 0
        all_arrays_alseis.append(out_image)

red_means_alseis=[]
green_means_alseis=[]
blue_means_alseis=[]
for tree in all_arrays_alseis:
    flatten_tree_red=tree[0,:,:].flatten()
    flatten_tree_green=tree[1,:,:].flatten()
    flatten_tree_blue=tree[2,:,:].flatten()
    red_means_alseis.append(flatten_tree_red[flatten_tree_red!=0].mean())
    green_means_alseis.append(flatten_tree_green[flatten_tree_green!=0].mean())
    blue_means_alseis.append(flatten_tree_blue[flatten_tree_blue!=0].mean())


plt.hist(red_means_alseis, bins=50, color='r', alpha=0.5)
plt.hist(green_means_alseis, bins=50, color='g', alpha=0.5)
plt.hist(blue_means_alseis, bins=50, color='b', alpha=0.5)
plt.show()

plt.hist(red_means, bins=50, color='r', alpha=0.5)
plt.hist(green_means, bins=50, color='g', alpha=0.5)
plt.hist(blue_means, bins=50, color='b', alpha=0.5)
plt.show()



fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot histograms for Alseis blackiana
axs[0].hist(red_means_alseis, bins=50, color='r', alpha=0.5)
axs[0].hist(green_means_alseis, bins=50, color='g', alpha=0.5)
axs[0].hist(blue_means_alseis, bins=50, color='b', alpha=0.5)
axs[0].set_title('Alseis blackiana')

# Plot histograms for Quararibea asterolepis
axs[1].hist(red_means, bins=50, color='r', alpha=0.5)
axs[1].hist(green_means, bins=50, color='g', alpha=0.5)
axs[1].hist(blue_means, bins=50, color='b', alpha=0.5)
axs[1].set_title('Quararibea asterolepis')

plt.tight_layout()
plt.show()




#denomination tags for subsets of the shapefiles 
tags= all_shapefiles["tag"].unique() 
tags_200= all_shapefiles[all_shapefiles["area"]>200]["tag"].unique() #tags_200 is the list of tags that have a crown area greater than 200 m2
tags_400= all_shapefiles[all_shapefiles["area"]>400]["tag"].unique() #tags_400 is the list of tags that have a crown area greater than 400 m2
tags_anacardium= all_shapefiles[all_shapefiles["Latin"]=="Anacardium excelsum"]["tag"].unique() #tags_anacardium is the list of tags that have a crown of Anacardium excelsum

k=3
thistag= all_shapefiles[all_shapefiles["tag"]==tags_400[k]]
print(thistag['tag'].unique())
thetag=thistag['tag'].unique()[0]
std=thistag["area"].std()
dpi = 80  
figsize = (1920 / dpi, 1080 / dpi)
plt.figure(figsize=figsize, dpi=dpi)
plt.plot(thistag["date"],thistag["area"])
plt.fill_between(thistag["date"],thistag["area"]-std,thistag["area"]+std,color='b',alpha=0.2)
plt.xticks(rotation=90)
if not os.path.exists(f"tree_{thetag}/"):
    os.makedirs(f"tree_{thetag}/")
plt.savefig(f"tree_{thetag}/crown_area_timeseries.png")


crowns_per_page = 12
crowns_plotted = 0
with PdfPages(f"tree_{thetag}/all_crowns_{thetag}.pdf") as pdf_pages:
    while crowns_plotted < len(orthos):
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        axes = axes.flatten()

        for i, (ortho_path, shp_path) in enumerate(zip(orthos[crowns_plotted:crowns_plotted + crowns_per_page], 
                                                       shps[crowns_plotted:crowns_plotted + crowns_per_page])):
            with rasterio.open(ortho_path) as src:
                shapefile = gpd.read_file(shp_path)
                crown = shapefile[shapefile["tag"]==tags_400[k]]
                bounds = crown.geometry.total_bounds
                box_crown_5 = box(bounds[0]-5, bounds[1]-5, bounds[2]+5, bounds[3]+5)
                out_image, out_transform = mask(src, [box_crown_5], crop=True)
                x_min, y_min = out_transform * (0, 0)
                xres, yres = out_transform[0], out_transform[4]
                transformed_geom = crown.geometry.apply(lambda geom: shapely.ops.transform(lambda x, y: ((x-x_min)/xres, (y-y_min)/yres), geom))
                axes[i].imshow(out_image.transpose((1, 2, 0))[:,:,0:3])
                transformed_geom.boundary.plot(ax=axes[i], edgecolor='red', linewidth=2)
                crowns_plotted += 1
                axes[i].axis('off')
            if crowns_plotted >= len(orthos):
                break

        plt.tight_layout()
        pdf_pages.savefig()
        plt.close(fig)




#so for this specific crown, lets decompose the color into histograms
all_arrays=[]
for i, thiscrown in enumerate(zip(orthos,shps)):
    if i >= 10:
        break
    print(thiscrown)
    with rasterio.open(thiscrown[0]) as src:
        shapefile=gpd.read_file(thiscrown[1])
        crown=shapefile[shapefile["tag"]==tags_400[k]]
        geom=crown["geometry"].iloc[0]
        out_image, out_transform = rasterio.mask.mask(src,[geom], crop=True)
        mask_nonzero = out_image != 0
        min_val = min(out_image[0,mask_nonzero[0,:,:]].min(), out_image[1,mask_nonzero[1,:,:]].min(), out_image[2,mask_nonzero[2,:,:]].min())
        max_val = max(out_image[0,mask_nonzero[0,:,:]].max(), out_image[1,mask_nonzero[1,:,:]].max(), out_image[2,mask_nonzero[2,:,:]].max())
        plt.hist(out_image[0,mask_nonzero[0,:,:]].flatten(), bins=50, color='r', alpha=0.5, range=(min_val, max_val))
        plt.hist(out_image[1,mask_nonzero[1,:,:]].flatten(), bins=50, color='g', alpha=0.5, range=(min_val, max_val))
        plt.hist(out_image[2,mask_nonzero[2,:,:]].flatten(), bins=50, color='b', alpha=0.5, range=(min_val, max_val))
        plt.show()

        #mask the range of colors falling to close to 0
        plt.imshow(out_image.transpose((1, 2, 0))[:,:,0])
        plt.show()

        #add to tuple
        all_arrays.append(out_image)


#timeseries of the 10 first distributions of blue in the crowns as whisker box plot
        
all_arrays=[]
data = []
labels = []

for i, thiscrown in enumerate(zip(orthos,shps)):
    if i >= 91:
        break
    print(thiscrown)
    with rasterio.open(thiscrown[0]) as src:
        shapefile=gpd.read_file(thiscrown[1])
        crown=shapefile[shapefile["tag"]==tags_400[k]]
        geom=crown["geometry"].iloc[0]
        out_image, out_transform = rasterio.mask.mask(src,[geom], crop=True)
        mask_nonzero = out_image != 0

        min_val = min(out_image[0,mask_nonzero[0,:,:]].min(), out_image[1,mask_nonzero[1,:,:]].min(), out_image[2,mask_nonzero[2,:,:]].min())
        max_val = max(out_image[0,mask_nonzero[0,:,:]].max(), out_image[1,mask_nonzero[1,:,:]].max(), out_image[2,mask_nonzero[2,:,:]].max())
        all_arrays.append(out_image)
        
        # Collect data for box plot
        data.append(out_image[2,mask_nonzero[2,:,:]].flatten())
        labels.append("_".join(thiscrown[1].split("\\")[-1].split("_")[0:3])
)  # Assuming "tag" is the name of the crown

# Create a box-and-whisker plot for all data
plt.boxplot(data, labels=labels)
plt.xticks(rotation=90)
plt.show()


import cv2
def normalize_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_l_channel = clahe.apply(l_channel)
    normalized_lab_image = cv2.merge((normalized_l_channel, a_channel, b_channel))
    normalized_rgb_image = cv2.cvtColor(normalized_lab_image, cv2.COLOR_LAB2RGB)
    return normalized_rgb_image

# List to store normalized RGB images
normalized_images = []

# Loop through the numpy array of RGB images
for i in range(len(all_arrays)):
    normalized_image = normalize_image(all_arrays[i][0:3,:,:].transpose(1, 2, 0))
    normalized_images.append(normalized_image)

# Prepare data for box plot after normalization
normalized_data = []

for i, img in enumerate(normalized_images):
    # Convert image back to original shape
    img = img.transpose(2, 0, 1)
    mask_nonzero = img != 0
    normalized_data.append(img[2,mask_nonzero[2,:,:]].flatten())

# Create a box-and-whisker plot for all normalized data
plt.boxplot(normalized_data, labels=labels)
plt.xticks(rotation=90)
plt.show()

len(all_arrays)
len(normalized_images)
# lets do a line plot with the mean of the blue channel for each crown
mean_blue = []
for img in normalized_images:
    mask_nonzero = img != 0
    mean_blue.append(img[2,mask_nonzero[2,:,:]].mean())

plt.plot(mean_blue)
plt.xticks(range(len(labels)), labels, rotation=90)
plt.show()


median_blue = [np.median(img[2, img[2,:,:]!=0]) for img in normalized_images]
plt.plot(median_blue)
plt.xticks(range(len(labels)), labels, rotation=90)
plt.show()


#save plotted crown


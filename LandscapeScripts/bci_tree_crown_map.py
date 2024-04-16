import os
import pandas as pd
import geopandas as gpd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import rasterio
import shapely
from shapely.geometry import box
from rasterio.mask import mask
import shapely.ops
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.merge import merge
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cv2
import numpy as np 
import math

#functions
def calculate_purple_score(masked_image, purple_colors):
    # Convert masked image to RGB
    rgb_image = masked_image.transpose((1, 2, 0))[:, :, 0:3]
    total_pixels = np.prod(rgb_image.shape[:2])
    black_pixels = np.sum(np.all(rgb_image == [0, 0, 0], axis=-1))
    total_pixels = total_pixels - black_pixels

    # Create a mask for each unique purple color and combine them
    purple_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    for color in purple_colors:
        color_mask = np.all(rgb_image == color, axis=-1)
        purple_mask = np.logical_or(purple_mask, color_mask)

    # Calculate the purple score
    purple_pixels = np.sum(purple_mask)
    purple_score = (purple_pixels / total_pixels) * 100

    print(f"Black pixels: {black_pixels}")
    print(f"Purple pixels: {purple_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Purple score: {purple_score}%")

    return purple_score,purple_pixels

#paths
jacaranda2_predicted_crowns=r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_03_18_crown_raw.shp"
jacaranda2=r"D:\BCI_tree_crown_map\orthomosaics\BCI_whole_2023_03_18_orthomosaic_jacaranda.tif"
jacaranda1=r"D:\BCI_tree_crown_map\orthomosaics\BCI_whole_2023_02_26_orthomosaic_jacaranda.tif"
BCI_outline=r"D:\BCI_tree_crown_map\aux_files\BCI_Outline.shp"
jacaranda2_blue_sample=r"D:\BCI_tree_crown_map\feature\jacaranda2_blue_sample.shp"
jacaranda1_blue_sample=r"D:\BCI_tree_crown_map\feature\jacaranda1_blue_sample.shp"
BCI_outline=gpd.read_file(BCI_outline)

#read aux files

#pixel library build
def process_sample(sample, raster):
    unique_colors = set()
    for idx, row in sample.iterrows():
        geometry= row["geometry"]
        box_geom = box(*geometry.bounds)
        with rasterio.open(raster) as src:
            masked, out_transform = mask(src, [box_geom], crop=True)
            rgb_image = masked.transpose((1, 2, 0))[:, :, 0:3]
            flattened = rgb_image.reshape(-1, rgb_image.shape[-1])
            unique_colors.update(map(tuple, np.unique(flattened, axis=0)))
    return unique_colors
def calculate_brightness(rgb_color):
    R, G, B = rgb_color
    return R + G + B

jacaranda2_blue_sample = gpd.read_file(jacaranda2_blue_sample)
jacaranda1_blue_sample = gpd.read_file(jacaranda1_blue_sample)
unique_colors_jacaranda2 = process_sample(jacaranda2_blue_sample, jacaranda2)
unique_colors_jacaranda1 = process_sample(jacaranda1_blue_sample, jacaranda1)
jacaranda_unique_colors = unique_colors_jacaranda2.union(unique_colors_jacaranda1)

jacaranda_unique_colors = {color for color in jacaranda_unique_colors if color != (0,0,0)}



size = int(math.ceil(math.sqrt(len(jacaranda_unique_colors))))
image = np.zeros((size, size, 3), dtype=np.uint8)
for i, color in enumerate(jacaranda_unique_colors):
    row = i // size
    col = i % size
    image[row, col] = color
plt.imshow(image)
plt.axis('off')
plt.show() 


#read BCI_whole_2023_03_18_crown_raw.shp which is the detectree stuff 
jacaranda2_crownmap=gpd.read_file(jacaranda2_predicted_crowns)
jacaranda2_crownmap_cleaned= gpd.overlay(jacaranda2_crownmap, BCI_outline, how="intersection")
jacaranda2_crownmap_cleaned.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_03_18_crownmap_jacaranda.gpkg", driver="GPKG")

with rasterio.open(jacaranda2) as src:
    for idx, row in jacaranda2_crownmap_cleaned.iterrows():
        geom = row["geometry"]
        masked, out_transform = mask(src, [geom], crop=True)
        purple_score, purple_pixels = calculate_purple_score(masked, trimmed_colors)
        # Add the purple_score to the crownmap GeoDataFrame
        jacaranda2_crownmap_cleaned.loc[idx, "purple_score"] = purple_score
        jacaranda2_crownmap_cleaned.loc[idx, "purple_pixels"] = purple_pixels
        print(f"Geometry {idx + 1}: Purple Score = {purple_score:.2f}%")

#plot distribution of non-zero purple scores
filtered_scores = jacaranda2_crownmap_cleaned["purple_pixels"][jacaranda2_crownmap_cleaned["purple_pixels"] > 0]
len(filtered_scores)
plt.hist(filtered_scores, bins=100)
plt.xlabel("Purple Score")
plt.ylabel("Frequency")
plt.title("Histogram of Purple Scores")
plt.show()

#plot individuals
crownmap=jacaranda2_crownmap_cleaned.sort_values("purple_score", ascending=True)
for idx, row in crownmap.iterrows():
    print(f"Geometry {idx + 1}: Purple Score = {row['purple_score']:.2f}%")
    if row['purple_score'] > 1:
        print(f"Geometry {idx + 1} is purple!")
        with rasterio.open(jacaranda2) as src:
            masked, out_transform = mask(src, [row["geometry"]], crop=True)
            show(masked, transform=out_transform)
            plt.show()

#filter the ones above 1
purple_crowns = jacaranda2_crownmap_cleaned[jacaranda2_crownmap_cleaned["purple_score"] > 1]
purple_crowns_pixels=jacaranda2_crownmap_cleaned[jacaranda2_crownmap_cleaned["purple_pixels"] > 1]
print(f"Number of purple crowns: {len(purple_crowns)}")
print(f"Number of purple crowns: {len(purple_crowns_pixels)}")
purple_crowns.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_03_18_crownmap_jacaranda.gpkg", driver="GPKG")
purple_crowns_pixels.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_03_18_crownmap_jacaranda_pixels.gpkg", driver="GPKG")





import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix
from skimage.feature import graycoprops

# Define the distance and angles for GLCM computation
distances = [1]  # You can adjust the distances as needed
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # You can adjust the angles as needed

# Initialize lists to store GLCM features
contrast_values = []
dissimilarity_values = []
homogeneity_values = []
energy_values = []
correlation_values = []

# Iterate over the masked image patches
with rasterio.open(jacaranda2) as src:
    for idx, row in jacaranda2_crownmap.iterrows():
        print(f"Processing crown {idx + 1}...")
        geom = row["geometry"]
        masked, out_transform = mask(src, [geom], crop=True)
        masked_gray = rgb2gray(masked.transpose((1, 2, 0))[:, :, 0:3])
        masked_gray.dtype=np.uint8
    # Compute GLCM
        glcm = graycomatrix(masked_gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        contrast= graycoprops(glcm, 'contrast')
        dissimilarity= graycoprops(glcm, 'dissimilarity')
        homogeneity= graycoprops(glcm, 'homogeneity')
        energy= graycoprops(glcm, 'energy')
        correlation= graycoprops(glcm, 'correlation')
        contrast_values.append(contrast)
        dissimilarity_values.append(dissimilarity)
        homogeneity_values.append(homogeneity)
        energy_values.append(energy)
        correlation_values.append(correlation)


    
# Add GLCM features to the crownmap GeoDataFrame
jacaranda2_crownmap['contrast'] = contrast_values
jacaranda2_crownmap['dissimilarity'] = dissimilarity_values
jacaranda2_crownmap['homogeneity'] = homogeneity_values
jacaranda2_crownmap['energy'] = energy_values
jacaranda2_crownmap['correlation'] = correlation_values





















#get statistics for each of the 2000 crowns
purple_crowns_pixels["orthomosaic"]= "BCI_whole_2023_03_18_orthomosaic_jacaranda.tif" 
import rasterstats
from rasterstats import zonal_stats
with rasterio.open(jacaranda2) as src:
    blue=src.read(3)
    affine = src.transform
stats_blue  = zonal_stats(purple_crowns_pixels, blue,affine=affine, stats=["mean", "median", "std", "min", "max", "count"])

purple_crowns_pixels["blue_mean"] = [stat["mean"] for stat in stats_blue]
purple_crowns_pixels["blue_median"] = [stat["median"] for stat in stats_blue]
purple_crowns_pixels["blue_std"] = [stat["std"] for stat in stats_blue]
purple_crowns_pixels["blue_min"] = [stat["min"] for stat in stats_blue]
purple_crowns_pixels["blue_max"] = [stat["max"] for stat in stats_blue]
purple_crowns_pixels["blue_count"] = [stat["count"] for stat in stats_blue]


plt.hist(purple_crowns_pixels["blue_mean"], bins=50)
plt.xlabel("Blue Mean")
plt.ylabel("Frequency")
plt.title("Histogram of Blue Mean")
plt.show()

purple_crowns_pixels=purple_crowns_pixels[purple_crowns_pixels["purple_pixels"] > 100]

purple_crowns_pixels.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_03_18_crownmap_jacaranda_pixels.gpkg", driver="GPKG")





import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming you have extracted features for each polygon and stored them in X
# Assuming you have labels for each polygon and stored them in y

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train classifier
clf.fit(X_train, y_train)

# Predict on testing data
y_pred = clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




















        






jacaranda1=r"D:\BCI_tree_crown_map\orthomosaics\BCI_whole_2023_03_18_orthomosaic_jacaranda.tif"
outline=r"D:\BCI_tree_crown_map\aux_files\BCI_Outline.shp"
outline=gpd.read_file(outline)
outline.plot()
plt.show()
with rasterio.open(jacaranda1) as src:
    masked, out_trans=mask(src, outline.geometry, crop=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": masked.shape[1],
        "width": masked.shape[2],
        "transform": out_trans,
    })
    with rasterio.open(r"D:\BCI_tree_crown_map\orthomosaics\BCI_whole_2023_03__orthomosaic_jacaranda_masked.tif", "w", **out_meta) as dest:
        dest.write(masked)


jacaranda1=r"D:\BCI_tree_crown_map\orthomosaics\BCI_whole_2023_02_26_orthomosaic_jacaranda_masked.tif"
with rasterio.open(jacaranda1) as src:
    data=src.read()
    plt.imshow(data.transpose((1, 2, 0))[:,:,0:3])
    plt.show()

#CALCULATE statistics color
def calculate_purple_score(masked_image):
    # Convert masked image to RGB
    rgb_image = masked_image.transpose((1, 2, 0))[:, :, 0:3]
    lower_purple = np.array([95,98, 137])  # Adjusted lower bounds for R, G, B
    upper_purple = np.array([191, 150, 214])  # Adjusted upper bounds for R, G, B
    purple_mask = np.all((rgb_image >= lower_purple) & (rgb_image <= upper_purple), axis=-1)
    purple_pixels = np.sum(purple_mask)
    total_pixels = np.prod(rgb_image.shape[:2])
    black_pixels = np.sum(np.all(rgb_image == [0, 0, 0], axis=-1))
    print(f"Black pixels: {black_pixels}")
    print(f"Purple pixels: {purple_pixels}")
    print(f"Total pixels: {total_pixels}")
    total_pixels= total_pixels-black_pixels
    purple_score = (purple_pixels / total_pixels) * 100
    return purple_score


path= r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_06_19_crownmap_version2.gpkg"
jacaranda2=r"D:\BCI_tree_crown_map\orthomosaics\BCI_whole_2023_03_18_orthomosaic_jacaranda.tif"



blue = r"D:\BCI_tree_crown_map\feature\jacaranca_blue_sample.shp"
blue2=r"D:\BCI_tree_crown_map\feature\jacaranca_blue_sample2.shp"

blue_sample2 = gpd.read_file(blue2)
#filter out parts of the sample where possibly the stretch type confused you and made you label the color wrongly. some of it  looks brown, reduce range of the greens
blue_sample = gpd.read_file(blue)

unique_colors = set()
for idx, row in blue_sample2.iterrows():
    geometry= row["geometry"]
    box_geom = box(*geometry.bounds)
    with rasterio.open(jacaranda1) as src:
        masked, out_transform = mask(src, [box_geom], crop=True)
        plt.imshow(masked.transpose((1, 2, 0))[:,:,0:3])
        plt.show()
        rgb_image = masked.transpose((1, 2, 0))[:, :, 0:3]
        flattened = rgb_image.reshape(-1, rgb_image.shape[-1])
        unique_colors.update(map(tuple, np.unique(flattened, axis=0)))

# Filter unique colors to get only the purple ones that have a blue component
purple_colors = [color for color in unique_colors if color[0] > color[1] and color[2] > color[1]]
unique_colors




#plot hist of red band green and blue
plt.hist(image[:,:,0].flatten(), bins=50, color='red', alpha=0.5, label='Red')
plt.hist(image[:,:,1].flatten(), bins=50, color='green', alpha=0.5, label='Green')
plt.hist(image[:,:,2].flatten(), bins=50, color='blue', alpha=0.5, label='Blue')
plt.legend()
plt.show()


path2=r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_02_26_crown_raw.shp"
crownmap=gpd.read_file(path2)

outline=r"D:\BCI_tree_crown_map\aux_files\BCI_Outline.shp"
outline=gpd.read_file(outline)



crownmap_edgesremoved = gpd.overlay(crownmap, outline, how="intersection")
crownmap_edgesremoved.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_02_26_crownmap_removed.gpkg", driver="GPKG")
crownmap.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_02_26_crownmap_removed.gpkg", driver="GPKG")
crownmap.columns
jacaranda1=r"D:\BCI_tree_crown_map\orthomosaics\BCI_whole_2023_02_26_orthomosaic_jacaranda.tif"


def calculate_purple_score(masked_image, purple_colors):
    # Convert masked image to RGB
    rgb_image = masked_image.transpose((1, 2, 0))[:, :, 0:3]
    total_pixels = np.prod(rgb_image.shape[:2])
    black_pixels = np.sum(np.all(rgb_image == [0, 0, 0], axis=-1))
    total_pixels = total_pixels - black_pixels

    # Create a mask for each unique purple color and combine them
    purple_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    for color in purple_colors:
        color_mask = np.all(rgb_image == color, axis=-1)
        purple_mask = np.logical_or(purple_mask, color_mask)

    # Calculate the purple score
    purple_pixels = np.sum(purple_mask)
    purple_score = (purple_pixels / total_pixels) * 100

    print(f"Black pixels: {black_pixels}")
    print(f"Purple pixels: {purple_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Purple score: {purple_score}%")

    return purple_score


for idx, row in crownmap.iterrows():
    geom = row["geometry"]
    with rasterio.open(jacaranda1) as src:
        masked, out_transform = mask(src, [geom], crop=True)
        plt.imshow(masked.transpose((1, 2, 0))[:,:,0:3])
        plt.show()


with rasterio.open(jacaranda1) as src:
        for idx, row in crownmap.iterrows():
            geom = row["geometry"]
            masked, out_transform = mask(src, [geom], crop=True)
            purple_score = calculate_purple_score(masked, purple_colors)
            # Add the purple_score to the crownmap GeoDataFrame
            crownmap.loc[idx, "purple_score"] = purple_score
            print(f"Geometry {idx + 1}: Purple Score = {purple_score:.2f}%")



bci_whole_v2=r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_06_19_crownmap_version2.gpkg"
bci_whole_v2=gpd.read_file(bci_whole_v2)
with rasterio.open(jacaranda2) as src:
    for idx, row in bci_whole_v2.iterrows():
        geom = row["geometry"]
        masked, out_transform = mask(src, [geom], crop=True)
        purple_score = calculate_purple_score(masked, purple_colors)
        # Add the purple_score to the crownmap GeoDataFrame
        bci_whole_v2.loc[idx, "purple_score"] = purple_score
        print(f"Geometry {idx + 1}: Purple Score = {purple_score:.2f}%")

import numpy as np

bci_whole_v2["sp_predicted"] = np.where(bci_whole_v2["purple_score"] > 0, "Jacaranda copaia", "Unknown")
bci_whole_v2["orthomosaic"]= "BCI_whole_2023_06_19_orthomosaic_jacaranda.tif"
bci_whole_v2.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_06_19_crownmap_version3.gpkg", driver="GPKG")
bci_whole_v2=gpd.read_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_06_19_crownmap_version3.gpkg")
bci_whole_v2["orthomosaic"]= "BCI_whole_2023_03_18_orthomosaic_jacaranda.tif"
bci_whole_v2.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_03_18_crownmap_version3.gpkg", driver="GPKG")

crownmap.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_02_26_crownmap_version2.gpkg", driver="GPKG")

crownmap=gpd.read_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_02_26_crownmap_version2.gpkg")
crownmap=_removed= gpd.overlay(crownmap, outline, how="intersection")
crownmap_filtered=crownmap[crownmap["purple_score"] > 0]
crownmap_filtered.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_02_26_crownmap_version2_filtered.gpkg", driver="GPKG")
crownmap_filtered=gpd.read_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_2023_02_26_crownmap_version2_filtered.gpkg")
crownmap_filtered["orthomosaic"]= "BCI_whole_2023_02_26_orthomosaic_jacaranda.tif"
crownmap_filtered["sp_predicted"] = np.where(crownmap_filtered["purple_score"] > 0, "Jacaranda copaia", "Unknown")

#bind both crownmaps
bci_whole_v2.crs
crownmap_filtered.crs=bci_whole_v2.crs
crownmap=pd.concat([bci_whole_v2, crownmap_filtered])
crownmap.to_file(r"D:\BCI_tree_crown_map\feature\BCI_whole_jacaranda.gpkg", driver="GPKG")

#read filtered bac

# Plot histogram of the filtered scores
filtered_scores = crownmap["purple_score"][crownmap["purple_score"] > 0]
plt.hist(filtered_scores, bins=50)
plt.xlabel("Purple Score")
plt.ylabel("Frequency")
plt.title("Histogram of Purple Scores")
plt.show()
# Print the crownmap with the newly added purple_score column
crownmap=crownmap.sort_values("purple_score", ascending=True)
for idx, row in crownmap.iterrows():
    print(f"Geometry {idx + 1}: Purple Score = {row['purple_score']:.2f}%")
    if row['purple_score'] > 0.2:
        print(f"Geometry {idx + 1} is purple!")
        with rasterio.open(jacaranda2) as src:
            masked, out_transform = mask(src, [row["geometry"]], crop=True)
            show(masked, transform=out_transform)
            plt.show()


#how many crowns are purple
purple_crowns = crownmap[crownmap["purple_score"] > 0]
print(f"Number of purple crowns: {len(purple_crowns)}")
#CHUNK TO PLOT THE CROWNS OF ALL THE TAGS FROM THE FOCAL SPECIES IN THE TIMESERIES OF THE 50HA PLOT IN BCI
crownmap2020=r"D:\BCI_50ha\crown_segmentation\2020_08_01_improved.shp"
crownmap2020=gpd.read_file(crownmap2020)

focal_species = [
    "Prioria copaifera",
    "Tabebuia rosea",
    "Handroanthus guayacan",
    "Jacaranda copaia",
    "Anacardium excelsum",
    "Platypodium elegans",
    "Zanthoxylum ekmanii",
    "Quararibea asterolepsis",
    "Ceiba pantandra",
    "Pseudobombax septenatum",
    "Virola surinamensis"
]

get_species = crownmap2020[crownmap2020["Latin"].isin(focal_species)]
print(get_species["tag"].unique())  

orthomosaic=r"D:\BCI_50ha\timeseries_local_alignment\BCI_50ha_2020_08_01_local.tif"
orthos = [os.path.join(r"D:\BCI_50ha\timeseries_local_alignment", f) for f in os.listdir(r"D:\BCI_50ha\timeseries_local_alignment") if f.endswith(".tif")]
shp_path = r"D:\BCI_50ha\crown_segmentation"
shps = [os.path.join(shp_path, f) for f in os.listdir(shp_path) if f.endswith("improved.shp")]


for k in get_species["tag"].unique():
    print(k)
    crowns_per_page = 12
    crowns_plotted = 0

    # Check if the file already exists
    if os.path.isfile(f"all_crowns_{k}.pdf"):
        print(f"Crown {k} has already been plotted.")
        continue

    with PdfPages(f"all_crowns_{k}.pdf") as pdf_pages:
        while crowns_plotted < len(orthos):
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            axes = axes.flatten()

            for i, (ortho_path, shp_path) in enumerate(zip(orthos[crowns_plotted:crowns_plotted + crowns_per_page], 
                                                        shps[crowns_plotted:crowns_plotted + crowns_per_page])):
                with rasterio.open(ortho_path) as src:
                    shapefile = gpd.read_file(shp_path)
                    crown = shapefile[shapefile["tag"]==k]
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


#
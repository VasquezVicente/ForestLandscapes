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

#PATHS
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
#data_path=r"C:\Users\Vicente\Documents\Data"
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
labels_path=r"timeseries/export_2025_03_05.csv"
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")

#list of orthomosaics
orthomosaic_list=os.listdir(orthomosaic_path)

#open gdf containing polygons
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")
#open df containing labels
labels=pd.read_csv(labels_path)
#merge labels and crowns, keeping only labeled ones
crowns_labeled= labels.merge(crowns[['area', 'score', 'tag', 'iou', 'geometry','polygon_id']],
                              left_on="polygon_id",
                                right_on="polygon_id",
                                  how="left")



#plot of most labeled species
species_counts = labels["latin"].value_counts()
top_species = species_counts.head(10)
plt.figure(figsize=(8, 8))
plt.pie(top_species, labels=top_species.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title("Most Labeled Species")
plt.show()

#plot individuals labeled by species
individual_counts = labels.groupby("latin")["globalId"].nunique().sort_values(ascending=False)
top_individuals = individual_counts.head(10)
plt.figure(figsize=(12, 10))
plt.bar(top_individuals.index, top_individuals.values, color='skyblue')
plt.xlabel("Species (Latin Name)")
plt.ylabel("Number of Unique Individuals Labeled")
plt.title("Top 10 Species by Labeled Individuals")
plt.xticks(rotation=15, ha="right")  # Rotate labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#QAQC
crowns_labeled.columns
crowns_labeled= crowns_labeled[crowns_labeled["segmentation"]=="good"]
crowns_labeled= crowns_labeled[crowns_labeled['isFlowering']=="no"]
unique_leafing_rows = crowns_labeled.drop_duplicates(subset=['leafing']).sort_values(by='leafing')


crowns_per_page = 12
crowns_plotted = 0

with PdfPages("plots/decidouss_example.pdf") as pdf_pages:
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()

    for i, (_, row) in enumerate(unique_leafing_rows.iterrows()):
        path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")

        with rasterio.open(path_orthomosaic) as src:
            bounds = row.geometry.bounds
            box_crown_5 = box(bounds[0]-5, bounds[1]-5, bounds[2]+5, bounds[3]+5)

            out_image, out_transform = mask(src, [box_crown_5], crop=True)
            x_min, y_min = out_transform * (0, 0)
            xres, yres = out_transform[0], out_transform[4]

            # Transform geometry
            transformed_geom = shapely.ops.transform(
                lambda x, y: ((x - x_min) / xres, (y - y_min) / yres),
                row.geometry
            )
 
            ax = axes[crowns_plotted % crowns_per_page]
            ax.imshow(out_image.transpose((1, 2, 0))[:, :, 0:3])
            ax.plot(*transformed_geom.exterior.xy, color='red', linewidth=2)
            ax.axis('off')

            # Add text label
            latin_name = row['latin']
            numeric_feature_1 = row['leafing']
            ax.text(5, 5, f"{latin_name}\nLeafing: {numeric_feature_1}",
                    fontsize=12, color='white', backgroundcolor='black', verticalalignment='top')

            crowns_plotted += 1

        # Save PDF and start a new page every 12 crowns
        if crowns_plotted % crowns_per_page == 0 or i == len(unique_leafing_rows) - 1:
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)

            # Create new figure for the next batch
            if i != len(unique_leafing_rows) - 1:  # Prevent unnecessary re-creation at end
                fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                axes = axes.flatten()


            elif transformed_geom.geom_type == "MultiPolygon":
                continue



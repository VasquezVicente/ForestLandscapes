import os
import geopandas as gpd
import rasterio
import pandas as pd
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.backends.backend_pdf import PdfPages
import shapely.ops
#load hura extracted
path= r"timeseries\dataset_extracted\ceiba.csv"
hura= pd.read_csv(path)

#load polygons

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")


species_subset=crowns[['geometry', 'polygon_id']].merge(hura, how='left', right_on='polygon_id', left_on='polygon_id')
individuals = species_subset['GlobalID'].dropna().unique()


def generate_leafing_pdf(unique_leafing_rows, output_pdf, orthomosaic_path, crowns_per_page=12, variables=[]):
    """
    Generates a PDF with deciduous crowns plotted.

    Parameters:
        output_pdf (str): Path to the output PDF file.
        unique_leafing_rows (GeoDataFrame): DataFrame containing crown geometries and metadata.
        orthomosaic_path (str): Path to the orthomosaic folder containing image files.
        crowns_per_page (int): Number of crowns to plot per page (default: 12).
        variables(tupple): must be numeric variables. 
    """
    crowns_plotted = 0

    with PdfPages(output_pdf) as pdf_pages:
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        axes = axes.flatten()

        for i, (_, row) in enumerate(unique_leafing_rows.iterrows()):
            path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")

            try:
                with rasterio.open(path_orthomosaic) as src:
                    bounds = row.geometry.bounds
                    box_crown_5 = box(bounds[0] - 5, bounds[1] - 5, bounds[2] + 5, bounds[3] + 5)

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
                    annotation_text = f"{row['latin']}\n"
                    for var in variables:
                        if var in row:
                            try:
                                val = float(row[var])
                                annotation_text += f"{var}: {val:.2f}\n"
                            except (ValueError, TypeError):
                                annotation_text += f"{var}: {row[var]}\n"
                    # Add text label
                    ax.text(5, 5, annotation_text.strip(),
                            fontsize=12, color='white', backgroundcolor='black', verticalalignment='top')
                    crowns_plotted += 1

            except Exception as e:
                print(f"Error processing {path_orthomosaic}: {e}")
                continue  # Skip the current iteration if an error occurs

            # Save PDF and start a new page every `crowns_per_page` crowns
            if crowns_plotted % crowns_per_page == 0 or i == len(unique_leafing_rows) - 1:
                plt.tight_layout()
                pdf_pages.savefig(fig)
                plt.close(fig)

                # Create new figure for the next batch
                if i != len(unique_leafing_rows) - 1:  # Prevent unnecessary re-creation at end
                    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                    axes = axes.flatten()
    print(f"PDF saved: {output_pdf}")


out_huras= r"//stri-sm01/ForestLandscapes/UAVSHARE/BCI_50ha_timeseries/videos/ceiba"
os.makedirs(out_huras, exist_ok=True)
ortho_path= os.path.join(data_path, 'orthomosaic_aligned_local')
            

for i in individuals:
    sub_indv= species_subset[species_subset['GlobalID']==i]
    sub_indv['date'] = sub_indv['date'].astype(str).str.replace("-", "_")
    
    out_path = os.path.join(out_huras, f"{i}.pdf")
    if not os.path.exists(out_path):
        generate_leafing_pdf(sub_indv,out_path, ortho_path,crowns_per_page=12,variables=["leafing","date"])
        print("finish one")
    else:
        print("already exist")

#timeseries functions
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
import os
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rasterio.mask import mask
from shapely.geometry import box
import shapely.ops
from statistics import mode
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
                            annotation_text += f"{var}: {row[var]:.2f}\n"
            
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

def customLeafing(leafing_values):
    values = list(leafing_values)
    sd_values = np.std(values)
    if len(values) == 1:
        return values[0]
    if len(values) >= 2 and sd_values <= 5:
        result= sum(values) / len(values)
        return result
    if len(values) >= 2 and sd_values > 5:
        try:
            reference_value = mode(values)
        except:
            reference_value = np.median(values)
        filtered_values = [v for v in values if abs(v - reference_value) <= 5]
        if filtered_values:
            result=sum(filtered_values) / len(filtered_values)
            print("third case", values, "result is:",result)
            return result
        else:
            result= None
            print("third case", values, "result is:",result)
            return result


def customLeafingNumeric(floweringN):
    values = list(floweringN)
    #easy stuff, if it is a maybe and is not confirmed, then replace all values == 2 to 0
    values = [0 if value == 2 else value for value in floweringN]
    sd_values = np.std(values)
    if len(values) == 1:
        return values[0]
    if len(values) >= 2 and sd_values <= 5:
        result= sum(values) / len(values)
        return result
    if len(values) >= 2 and sd_values > 5:
        try:
            reference_value = mode(values)
        except:
            reference_value = np.median(values)
        filtered_values = [v for v in values if abs(v - reference_value) <= 5]
        if filtered_values:
            result=sum(filtered_values) / len(filtered_values)
            print("third case", values, "result is:",result)
            return result
        else:
            result= None
            print("third case", values, "result is:",result)
            return result


def customFlowering(floweringValues): 
    floweringValues=list(floweringValues)
    if len(floweringValues)==1:
        return floweringValues[0]
    if all(value == floweringValues[0] for value in floweringValues):
            return floweringValues[0]
    if "maybe" in floweringValues:
        return "maybe"
    if "yes" in floweringValues and "no" in floweringValues:
        return "maybe"
    

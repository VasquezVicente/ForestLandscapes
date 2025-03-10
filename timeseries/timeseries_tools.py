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

def generate_leafing_pdf(output_pdf, unique_leafing_rows, orthomosaic_path, crowns_per_page=12):
    """
    Generates a PDF with deciduous crowns plotted.

    Parameters:
        output_pdf (str): Path to the output PDF file.
        unique_leafing_rows (GeoDataFrame): DataFrame containing crown geometries and metadata.
        orthomosaic_path (str): Path to the orthomosaic folder containing image files.
        crowns_per_page (int): Number of crowns to plot per page (default: 12).
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
                    latin_name = row['latin']
                    numeric_feature_1 = row['leafing']
                    numeric_feature_2 = row['date']
                    ax.text(5, 5, f"{latin_name}\nLeafing: {numeric_feature_1}\date: {numeric_feature_2}",
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


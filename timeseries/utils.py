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
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from shapely.ops import unary_union

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

def customFloweringNumeric(floweringN):
    values = list(floweringN)
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

def calculate_glcm_features(image, window_size=5, angles=[0, 45, 90, 135]):
    glcm_features = []
    for angle in angles:
        # Calculate the GLCM for the specified angle
        glcm = graycomatrix(img_as_ubyte(image), distances=[1], angles=[np.deg2rad(angle)], symmetric=True, normed=True) 
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        glcm_features.append(correlation)
    return glcm_features  


def create_overlap_density_map(gdf, grid_size=0.5):
    """Create a density map showing polygon overlap intensity - y increases downward (for origin='upper')."""
    bounds = gdf.total_bounds
    x_min, y_min, x_max, y_max = bounds

    width = int(np.ceil((x_max - x_min) / grid_size))
    height = int(np.ceil((y_max - y_min) / grid_size))

    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    density_matrix = np.zeros((height, width), dtype=np.int32)

    for geom in gdf['geometry']:
        if geom.is_valid:
            raster = rasterize(
                [geom],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                default_value=1,
                dtype=np.int32
            )
            density_matrix += raster

    # y_coords now increases from y_min (top) to y_max (bottom)
    x_coords = np.arange(x_min, x_max, grid_size)
    y_coords = np.arange(y_min, y_max, grid_size)
    # Reverse rows so first row is y_min (bottom)
    density_matrix = density_matrix[::-1, :]
    return x_coords, y_coords, density_matrix

def create_consensus_polygon(x_coords, y_coords, density_matrix, threshold=0.5):
    """
    Create a consensus polygon from density matrix
    
    Parameters:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates  
        density_matrix: 2D array of overlap densities
        threshold: Threshold for normalized density (0-1)
    
    Returns:
        Consensus polygon or None
    """
    # Normalize density matrix to 0-1 range
    if np.max(density_matrix) > 0:
        density_norm = (density_matrix - np.min(density_matrix)) / (np.max(density_matrix) - np.min(density_matrix))
    else:
        return None
    
    # Mask values below threshold
    density_norm[density_norm < threshold] = np.nan
    
    # Create polygons for high-density grid cells
    grid_size = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    polygons_from_density = []
    
    for i in range(density_norm.shape[0]):
        for j in range(density_norm.shape[1]):
            if not np.isnan(density_norm[i, j]):
                # Get grid cell coordinates
                x = x_coords[j]
                y = y_coords[i]
                
                # Create square polygon for this grid cell
                square = Polygon([
                    (x, y),
                    (x + grid_size, y),
                    (x + grid_size, y + grid_size),
                    (x, y + grid_size)
                ])
                polygons_from_density.append(square)
    
    # Union all squares into single polygon
    if polygons_from_density:
        consensus_polygon = unary_union(polygons_from_density)
        return consensus_polygon
    else:
        return None

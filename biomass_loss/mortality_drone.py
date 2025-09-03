import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.mask import mask
import rasterio
from shapely.geometry import box
from matplotlib.backends.backend_pdf import PdfPages
import shapely.ops
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
    
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
            path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date'].strftime('%Y_%m_%d')}",f"BCI_50ha_{row['date'].strftime('%Y_%m_%d')}_tile_{int(row['tile_id'])-1}.tif")
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
                    annotation_text = ""
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

# Define paths
othomosaic_tiles_path= r"\\stri-sm01\ForestLandscapes\UAVSHARE\biomass_losses_Luisa\TreeApproach\DroneImageryAligment\Product_tiles_vertical"
crown_shapefile_path= r"\\stri-sm01\ForestLandscapes\UAVSHARE\biomass_losses_Luisa\TreeApproach\crowns_segmentation\crown_segmentation_by_tiles\crownmap_predicted"
crown_segmentation_check_path = r"\\stri-sm01\ForestLandscapes\UAVSHARE\biomass_losses_Luisa\TreeApproach\crowns_segmentation\crown_segmentation_check"
os.makedirs(crown_segmentation_check_path, exist_ok=True)
#list all the shp files in the folder, walk all the way through subfolders
shp_files = []
for root, dirs, files in os.walk(crown_shapefile_path):
    for f in files:
        if f.endswith('.shp'):
            shp_files.append(os.path.join(root, f))


shp_df= pd.DataFrame(shp_files, columns=['shp_path'])
shp_df['file_name'] = shp_df['shp_path'].apply(lambda x: os.path.basename(x))
shp_df['tile_id'] = shp_df['file_name'].apply(lambda x: x.split('_')[5]) 
shp_df['date'] = shp_df['file_name'].apply(lambda x: '_'.join(x.split('_')[1:4]))

##unique tiles
unique_tiles = shp_df['tile_id'].unique()

all_polygons = []
for tile in unique_tiles:
    print(f"Processing tile: {tile}")
    tile_shp_files = shp_df[shp_df['tile_id'] == str(tile)]
    for idx, row in tile_shp_files.iterrows():
        gdf = gpd.read_file(row['shp_path'])
        gdf['date'] = row['date']
        gdf['tile_id'] = row['tile_id']
        all_polygons.append(gdf)
        

master_gdf = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=all_polygons[0].crs)
master_gdf['date'] = master_gdf['date'].apply(lambda x: pd.to_datetime(x, format='%Y_%m_%d'))
master_gdf['area'] = master_gdf['geometry'].area
unique_globalids = master_gdf['GlobalID'].unique()

all_gid_polygons = []
for gid in unique_globalids:
    subset = master_gdf[master_gdf['GlobalID'] == gid]
    grid_size = 0.1
    x_coords, y_coords, density_matrix = create_overlap_density_map(subset, grid_size=grid_size)
    consensus_polygon = create_consensus_polygon(x_coords, y_coords, density_matrix)

    #this now plots correctly aligned
    fig, ax = plt.subplots(figsize=(10, 10))
    extent = [subset.total_bounds[0], subset.total_bounds[2], subset.total_bounds[1], subset.total_bounds[3]]
    im = ax.imshow(density_matrix, extent=extent, origin='lower', cmap='hot', alpha=0.7)
    subset['geometry'].plot(ax=ax, edgecolor='white', facecolor='none', alpha=0.8, linewidth=1)
    #plot concensus polygon if it exists
    if consensus_polygon is not None:
        gpd.GeoSeries(consensus_polygon).plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2)
    plt.colorbar(im, ax=ax, label='Overlap Count')
    plt.title(f"Density Map with Polygons - {gid}")
    plt.show()

    # Calculate Hausdorff distances and replace high-distance polygons
    if consensus_polygon is not None:
        # First loop: calculate Hausdorff distances
        for idx, row in subset.iterrows():
            poly = row['geometry']
            distance = poly.hausdorff_distance(consensus_polygon)
            subset.loc[idx, 'hausdorff_distance'] = distance
            print(f"Polygon has Hausdorff distance: {distance:.2f}")

        # Calculate statistics for outlier detection
        distances = subset['hausdorff_distance']
        mean_distance = distances.mean()
        std_distance = distances.std()

        final_threshold= mean_distance + 1 * std_distance
        
        # Use the more conservative threshold

        print(f"Mean Hausdorff distance for {gid}: {mean_distance:.2f}")
        print(f"Standard deviation: {std_distance:.2f}")
        print(f"Z-score threshold (μ + 1.5σ): {final_threshold:.2f}")
        print(f"Final threshold used: {final_threshold:.2f}")
        
        # Second loop: replace only significant outliers
        replacements_made = 0
        for idx, row in subset.iterrows():
            if row['hausdorff_distance'] > final_threshold:
                print(f"Polygon with GlobalID {gid} has high Hausdorff distance: {row['hausdorff_distance']:.2f} - REPLACING")
                subset.loc[idx, 'geometry'] = consensus_polygon
                replacements_made += 1
        
        print(f"Replaced {replacements_made}/{len(subset)} polygons ({replacements_made/len(subset)*100:.1f}%)")

    generate_leafing_pdf(subset, os.path.join(crown_segmentation_check_path, f"{gid}.pdf"), othomosaic_tiles_path, crowns_per_page=12, variables=['hausdorff_distance', 'date'])
    subset['area'] = subset['geometry'].area
    all_gid_polygons.append(subset)

final_gdf = gpd.GeoDataFrame(pd.concat(all_gid_polygons, ignore_index=True), crs=subset.crs)

for gid in unique_globalids:
    gid_subset = final_gdf[final_gdf['GlobalID'] == gid]
    distances = gid_subset['hausdorff_distance']
    mean_distance = distances.mean()
    std_distance = distances.std()
    final_threshold = mean_distance + 1 * std_distance

    print(f"Mean Hausdorff distance for {gid}: {mean_distance:.2f}")
    print(f"Standard deviation: {std_distance:.2f}")
    print(f"Z-score threshold (μ + 1.5σ): {final_threshold:.2f}")
    print(f"Final threshold used: {final_threshold:.2f}")

    replacements_made = 0
    for idx, row in gid_subset.iterrows():
        if row['hausdorff_distance'] > final_threshold:
            print(f"Polygon with GlobalID {gid} has high Hausdorff distance: {row['hausdorff_distance']:.2f} - FLAGGING BAD")
            final_gdf.loc[idx, 'flag'] = 'BAD'
            replacements_made += 1
        else:
            final_gdf.loc[idx, 'flag'] = 'GOOD'

final_gdf['area'] = final_gdf['geometry'].area
final_gdf.to_file(os.path.join(crown_segmentation_check_path, "final_crown_segmentation_checked.shp"))

len(unique_globalids), len(final_gdf), final_gdf['flag'].value_counts()
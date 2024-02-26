import os
import geopandas as gpd
import pandas as pd
import rasterio 
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
from skimage import exposure
from skimage.transform import rescale
from scipy.ndimage import zoom

#utility modules
import matplotlib.pyplot as plt
import os
import pandas as pd
import rasterio
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import box as box1
from matplotlib.patches import Rectangle
import time 
#AI
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
import torch
#rasterio functions
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio import windows
from rasterio.plot import show

current_directory = os.getcwd()
print(current_directory)

#50ha path
shape_50ha=r"D:\BCI50ha_lidar\aux_files\BCI_Plot_50ha.shp"
shapefile = gpd.read_file(shape_50ha)
shapefile.to_crs(epsg=32617, inplace=True)
print(shapefile.crs)
shapefile.plot()
plt.show()

MODEL_TYPE = "vit_h"
checkpoint = r"C:\Users\VasquezV\repo\crown-segment\models\sam_vit_h_4b8939.pth"
device = 'cuda'
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
#sam.to(device=device)  #requires cuda cores
mask_predictor = SamPredictor(sam)


#functions
def tile_ortho(sub, tile_size, buffer, output_folder):
    with rasterio.open(sub) as src:
        bounds = src.bounds
        xmin, ymin, xmax, ymax = bounds
        if tile_size <= 0:
            raise ValueError("tile_size must be greater than zero.")      
        x_range = xmax - xmin
        y_range = ymax - ymin
        x_tiles = int(np.ceil(x_range / tile_size))
        y_tiles = int(np.ceil(y_range / tile_size))
        x_residual = x_range % tile_size
        y_residual = y_range % tile_size
        if x_residual > 0:
            tile_size_x = tile_size + x_residual / x_tiles
        else:
            tile_size_x = tile_size
        if y_residual > 0:
            tile_size_y = tile_size + y_residual / y_tiles
        else:
            tile_size_y = tile_size
        if x_residual > 0 or y_residual > 0:
            print(f"Warning: Adjusted tile size used for residual coverage - X: {tile_size_x}, Y: {tile_size_y}")
        xmins = np.arange(xmin, (xmax - tile_size_x + 1), tile_size_x)
        xmaxs = np.arange((xmin + tile_size_x), xmax + 1, tile_size_x)
        ymins = np.arange(ymin, (ymax - tile_size_y + 1), tile_size_y)
        ymaxs = np.arange((ymin + tile_size_y), ymax + 1, tile_size_y)
        X, Y = np.meshgrid(xmins, ymins)
        Xmax, Ymax = np.meshgrid(xmaxs, ymaxs)
        gridInfo = pd.DataFrame({
            'xmin': X.flatten(),
            'ymin': Y.flatten(),
            'xmax': Xmax.flatten(),
            'ymax': Ymax.flatten(),
        })
        print(gridInfo)
    with rasterio.open(sub) as src:
        for idx, row in gridInfo.iterrows():
            geom2 = box1(row['xmin']-buffer, row['ymin']-buffer, row['xmax']+buffer, row['ymax']+buffer)
            out_image, out_transform = rasterio.mask.mask(src, [geom2], crop=True)
            # Update metadata for the output raster
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            output_filename = f"output_raster_{idx}.tif"
            filename=os.path.join(output_folder,output_filename)
            with rasterio.open(filename, "w", **out_meta) as dest:
                dest.write(out_image)
def crown_segment(tile_folder,shp,output_shapefile):
    all=[]
    tiles= os.listdir(tile_folder)
    for tile in tiles:
        sub=os.path.join(tile_folder,tile)
        with rasterio.open(sub) as src:
            data=src.read()
            transposed_data=data.transpose(1,2,0)
            crs=src.crs
            affine_transform = src.transform 
            bounds=src.bounds
            main_box= box1(bounds[0],bounds[1],bounds[2],bounds[3])
        crowns=gpd.read_file(shp)
        crowns= crowns.to_crs(crs)
        mask = crowns['geometry'].within(main_box)
        test_crowns = crowns.loc[mask]

        print("starting box transformation from utm to xy")
        boxes=[]
        for index, row in test_crowns.iterrows():
            if isinstance(row.geometry, MultiPolygon):
                multi_polygon = row.geometry
                polygons = []
                for polygon in multi_polygon.geoms:
                    polygons.append(polygon)
                largest_polygon = max(polygons, key=lambda polygon: polygon.area)
                bounds = largest_polygon.bounds
                boxes.append(bounds)
                print("found one multipolygon for tag", row['tag'])
            else:
                bounds = row.geometry.bounds
                boxes.append(bounds)
        box_mod=[]
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            x_pixel_min, y_pixel_min = ~affine_transform * (xmin, ymin)
            x_pixel_max, y_pixel_max = ~affine_transform * (xmax, ymax)
            trans_box=[x_pixel_min,y_pixel_max,x_pixel_max,y_pixel_min]
            box_mod.append(trans_box)
        if not box_mod:
            print(f"No valid boxes found for tile {tile}. Skipping.")
            continue
        print("The tile contains", len(box_mod), "polygons")

        input_boxes=torch.tensor(box_mod, device=mask_predictor.device)
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, transposed_data[:,:,:3].shape[:2])
        print("about to set the image")
        mask_predictor.set_image(transposed_data[:,:,:3])
        masks, scores, logits= mask_predictor.predict_torch(point_coords=None,
            point_labels=None,boxes=transformed_boxes, multimask_output=True,)
        
        print("finish predicting now getting the utms for transformation")
        height, width, num_bands = transposed_data.shape
        utm_coordinates_and_values = np.empty((height, width, num_bands + 2))
        utm_transform = src.transform
       
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        utm_x, utm_y = rasterio.transform.xy(utm_transform, y_coords, x_coords)
        utm_coordinates_and_values[..., 0] = utm_x
        utm_coordinates_and_values[..., 1] = utm_y
        utm_coordinates_and_values[..., 2:] = transposed_data[..., :num_bands]

        all_polygons=[]
        for idx, (thisscore, thiscrown) in enumerate(zip(scores, masks)):
            maxidx=thisscore.tolist().index(max(thisscore.tolist()))
            thiscrown = thiscrown[maxidx]
            score=scores[1].tolist()[thisscore.tolist().index(max(thisscore.tolist()))]   
            mask = thiscrown.squeeze()
            utm_coordinates = utm_coordinates_and_values[:, :, :2]
            mask_np = mask.cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            areas = []
            for contour in contours:
                contour_coords = contour.squeeze().reshape(-1, 2)
                contour_utm_coords = utm_coordinates[contour_coords[:, 1], contour_coords[:, 0]]
                if len(contour_utm_coords) >= 3:
                    polygon = Polygon(contour_utm_coords)
                    area = polygon.area
                    polygons.append(polygon)
                    areas.append(area)       
            if len(areas) == 0:
                print(f"No valid areas found for this crown. Skipping.")
                continue  
            largest_index = np.argmax(areas)
            gdf = gpd.GeoDataFrame(geometry=[polygons[largest_index]])
            gdf['area'] = areas[largest_index]
            gdf['score'] = score 
            gdf.crs = src.crs
            tag_value = test_crowns.iloc[idx]['tag']
            global_id= test_crowns.iloc[idx]['GlobalID']
            gdf['tag']=tag_value
            gdf['GlobalID']=global_id
            all_polygons.append(gdf)
        print("finish transforming back to utm")
        print(len(all_polygons),"crowns segmented")
        all.append(all_polygons)
        progress= len(all)/len(tiles)
        print(progress)
    final_gdfs = []
    for polygons_gdf_list in all:
        combined_gdf = gpd.GeoDataFrame(pd.concat(polygons_gdf_list, ignore_index=True), crs=src.crs)
        final_gdfs.append(combined_gdf)
    final_gdf = gpd.GeoDataFrame(pd.concat(final_gdfs, ignore_index=True), crs=src.crs)
    final_gdf.to_file(output_shapefile)
def crown_avoid(dir_out2):
    crown_avoidance= gpd.read_file(dir_out2)
    crown_avoidance['geometry'] = crown_avoidance.geometry.buffer(0)

    overlapping_crowns = crown_avoidance[crown_avoidance.apply(lambda row: crown_avoidance.loc[crown_avoidance.index != row.name].within(row.geometry).any(), axis=1)]
    if not overlapping_crowns.empty:
        print("There are overlapping crowns.")
        overlaps_exist = True
    else:
        print("There are no overlapping crowns.")
        overlaps_exist = False

    #ideally while loop should stop when there is no more overlapping crowns, however it doest not stop
    #The counter is a temporary solution for crown avoidance but it is not ideal
    counter = 0
    while overlaps_exist:
        counter += 1
        for idx, crown in crown_avoidance.iterrows():
                    adjacent = crown_avoidance[(crown_avoidance.geometry.intersects(crown['geometry'])) & (crown_avoidance.index != idx)]
                    if adjacent.empty:
                        continue
                    else:
                        for adj_idx, adj_crown in adjacent.iterrows():
                            overlap_area_a_to_b = crown['geometry'].intersection(adj_crown['geometry']).area
                            overlap_area_b_to_a = adj_crown['geometry'].intersection(crown['geometry']).area
                            overlap_percentage_a_to_b = (overlap_area_a_to_b / crown['geometry'].area) * 100
                            overlap_percentage_b_to_a = (overlap_area_b_to_a / adj_crown['geometry'].area) * 100

                            if overlap_percentage_a_to_b > overlap_percentage_b_to_a:
                                # Adjust the areas
                                crown_avoidance.loc[idx, 'geometry'] = crown['geometry'].union(adj_crown['geometry'].intersection(crown['geometry']))
                                crown_avoidance.loc[adj_idx, 'geometry'] = adj_crown['geometry'].difference(crown['geometry'].intersection(adj_crown['geometry']))
                                print("adjusted crown case a", idx)
                            elif overlap_percentage_b_to_a > overlap_percentage_a_to_b:
                                # Adjust the areas
                                crown_avoidance.loc[adj_idx, 'geometry'] = adj_crown['geometry'].union(crown['geometry'].intersection(adj_crown['geometry']))
                                crown_avoidance.loc[idx, 'geometry'] = crown['geometry'].difference(adj_crown['geometry'].intersection(crown['geometry']))
                                print("adjusted crown case b", adj_idx)
                            else:
                                continue
                    

        for index, row in crown_avoidance.iterrows():
                            if isinstance(row.geometry, MultiPolygon):
                                print("found one multipolygon")
                                multi_polygon = row.geometry 
                                polygons = []
                                for polygon in multi_polygon.geoms:
                                    polygons.append(polygon)
                                largest_polygon = max(polygons, key=lambda polygon: polygon.area)
                                crown_avoidance.at[index, 'geometry'] = largest_polygon
                            else:
                                continue
        
        overlapping_crowns = crown_avoidance[crown_avoidance.apply(lambda row: crown_avoidance.loc[crown_avoidance.index != row.name].within(row.geometry).any(), axis=1)]

        if overlapping_crowns.empty:
            print("There are no overlapping crowns.")
            overlaps_exist = False
        else:
            print("There are overlapping crowns.")
            overlaps_exist = True
        
        all_adjacents_empty = all(crown_avoidance[(crown_avoidance.geometry.intersects(crown['geometry'])) & (crown_avoidance.index != idx)].empty for idx, crown in crown_avoidance.iterrows())
        if counter >= 10:
            print("Reached 10 iterations, terminating loop.")
            break
    return crown_avoidance
  
crownmaps_path = r"D:\crown_maps"
local_aligment= r"D:\BCI_50ha_timeseries_local_alignment" 

list_orthomosaics=[os.path.join(local_aligment, f) for f in os.listdir(local_aligment) if f.endswith('.tif')]
list_orthomosaics.sort()

reference_crownmap = gpd.read_file(os.path.join(crownmaps_path, "BCI_50ha_2020_08_01_crownmap_improved", "BCI_50ha_2020_08_01_crownmap_improved.shp"))
reference_orthomosaic = list_orthomosaics[24]
print("reference orthomosaic:", reference_orthomosaic)



#plot both for visualization purposes
with rasterio.open(reference_orthomosaic) as src:
    orthomosaic = src.read()
    p2, p98 = np.percentile(orthomosaic, (5, 95))
    orthomosaic_stretched = exposure.rescale_intensity(orthomosaic, in_range=(p2, p98))
    fig, ax = plt.subplots(figsize=(10, 10))
    rasterio.plot.show(orthomosaic_stretched, transform=src.transform, ax=ax)
    # Plot the geometries with no fill color
    boundaries = gpd.GeoDataFrame(reference_crownmap['Mnemonic'], geometry=reference_crownmap.geometry.boundary)
    boundaries.plot(ax=ax, column='Mnemonic', legend=False, cmap='inferno',linewidth=0.6)
    shapefile.geometry.boundary.plot(ax=ax, color=None, edgecolor='k', linewidth=0.5)
    # Plot the boundaries
    plt.title('Reference Crown Map BCI 50ha 2020-08-01')
    fig.savefig('vis/reference_crownmap.png', dpi=300, bbox_inches='tight')


#summary of the crown map
total_species= len(reference_crownmap['Mnemonic'].unique())
total_tags= len(reference_crownmap['tag'].unique())
total_area= reference_crownmap['geometry'].area.sum() #meter squared 
total_area_50ha= shapefile['geometry'].area.sum()
total_percentage= (total_area/total_area_50ha)*100
reference_crownmap_summary = pd.DataFrame({'Total Species': int(total_species), 'Total Tags': int(total_tags),
                                            'Mapped Area (m2)': round(total_area,0), 'Total Area 50ha (m2)': int(total_area_50ha)}, index=[0])
df = reference_crownmap_summary.T
fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
# Adjust the layout
fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc = 'center', loc='center')
table.auto_set_font_size(False) 
table.set_fontsize(10)
fig.savefig('vis/reference2020summary.png', dpi=300, bbox_inches='tight')
# Plot the highest abundance species
species_abundance = reference_crownmap['Latin'].value_counts()
species_abundance = species_abundance.nlargest(10).to_frame().reset_index()
#plot table
fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
table = ax.table(cellText=species_abundance.values, colLabels=species_abundance.columns, cellLoc = 'center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
fig.savefig('vis/reference2020speciesabundance.png', dpi=300, bbox_inches='tight')



#lets plot the top species in the crown map 4 polygons with the higuest area, we can use rasterio to mask the orthomosaic and plot them in a 4x4 latyouts
# Get the species with the second highest count
second_species = reference_crownmap['Latin'].value_counts().nlargest(4).index[3]
second_species_gdf = reference_crownmap[reference_crownmap['Latin'] == second_species]
second_species_gdf = second_species_gdf.sort_values('crownArea', ascending=False).head(9)
print(second_species_gdf["Latin"])

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
for i, row in enumerate(second_species_gdf.itertuples()):
     geom= row.geometry.bounds
     bbox= box(geom[0]-3,geom[1]-3,geom[2]+3,geom[3]+3)
     with rasterio.open(reference_orthomosaic) as src:
        out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True)
        out_meta = src.meta
        xs, ys = row.geometry.exterior.xy
        pixel_coords = [(x, y) * ~out_transform for x, y in zip(xs, ys)]
        ax = axs[i // 3, i % 3]
        ax.imshow(out_image.transpose(1, 2, 0)[:, :, :3])
        ax.plot(*zip(*pixel_coords), color='red')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.subplots_adjust(top=0.9)
fig.suptitle(f'Top 9 instances of {second_species}', fontsize=16, y=0.95)
plt.savefig(f'vis/{second_species}_9.png', bbox_inches='tight')
plt.show()



#read in the crown map 2020 
crown2020_path=os.path.join(crownmaps_path, "BCI_50ha_2020_08_01_crownmap_improved", "BCI_50ha_2020_08_01_crownmap_improved.shp")
crownmap_2020 = gpd.read_file(crown2020_path)

#read in the local alignment orthomosaic
orthomosaic_list = [os.path.join(local_aligment, f) for f in os.listdir(local_aligment) if f.endswith('.tif')]
orthomosaic_list.sort()


#align the previous date
num=23
tile_folder=os.path.join(local_aligment, "tiles",orthomosaic_list[num].split("\\")[2].replace("_local.tif","_tile"))
if not os.path.exists(tile_folder):
    os.makedirs(tile_folder)
tile_ortho(orthomosaic_list[num],100,20,tile_folder)

#vis the tiles and polygons
list_tiles= [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]

with rasterio.open(list_tiles[10]) as src:
    src_data = src.read()
    bounds = src.bounds
    src_transform = src.transform
    main_box = box1(bounds[0],bounds[1],bounds[2],bounds[3])
    mask = reference_crownmap['geometry'].within(main_box)
    test_crowns = reference_crownmap.loc[mask]
    fig, ax = plt.subplots()
    show(src_data[:3,:,:], transform=src_transform,ax=ax)
    test_crowns.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
    plt.show()
    #save
    fig.savefig('vis/tile10_crownmap.png', dpi=300, bbox_inches='tight')

with rasterio.open(list_tiles[10]) as src:
    src_data = src.read()
    bounds = src.bounds
    src_transform = src.transform
    main_box = box1(bounds[0],bounds[1],bounds[2],bounds[3])
    mask = crownmap_2020['geometry'].within(main_box)
    test_crowns = crownmap_2020.loc[mask]

    # Replace all geometries with their bounding boxes
    test_crowns['geometry'] = test_crowns['geometry'].apply(lambda geom: box1(*geom.bounds))

    fig, ax = plt.subplots()
    show(src_data[:3,:,:], transform=src_transform,ax=ax)
    test_crowns.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
    plt.show()

    #save
    fig.savefig('vis/tile10_crowmaop_boxes.png', dpi=300, bbox_inches='tight')


src_data.shape
out_shp_folder= os.path.join(r"D:\crown_maps\crown_segmentation",orthomosaic_list[num].split("\\")[2].replace("_local.tif","_segmented"))
if not os.path.exists(out_shp_folder):
    os.makedirs(out_shp_folder)
output_shp= os.path.join(out_shp_folder,orthomosaic_list[num].split("\\")[2].replace("_local.tif","_segmented.shp"))
crown_segment(tile_folder,crown2020_path,output_shp)



for ortho in reversed(orthomosaic_list[:24]):
     print(ortho)
     if not os.path.exists(os.path.join(r"D:\BCI_50ha_timeseries_local_alignment\tiles","_".join(os.path.basename(ortho).split("_")[2:5]))):
         os.makedirs(os.path.join(r"D:\BCI_50ha_timeseries_local_alignment\tiles", os.path.basename(ortho).replace(".tif","")))
     tile_ortho(ortho,100,20,os.path.join(r"D:\BCI_50ha_timeseries_local_alignment\tiles","_".join(os.path.basename(ortho).split("_")[2:5])))
     print("finish tiling orthomosaic")
     input_orthomosaic= os.path.join(r"D:\BCI_50ha_timeseries_local_alignment\tiles","_".join(os.path.basename(ortho).split("_")[2:5]))
     output_shp= os.path.join(r"D:\crown_maps\crown_segmentation", "_".join(os.path.basename(ortho).split("_")[2:5])+".shp")
     crown_segment(input_orthomosaic,crown2020_path,output_shp)
     print("finish instance segmentation")

     crownmap=gpd.read_file(output_shp)
     crownmap2020_tag = crownmap.sort_values('score', ascending=False)
     crownmap2020_tag = crownmap.drop_duplicates('GlobalID', keep='first')

     dir_out2= os.path.join(r"D:\crown_maps\crown_segmentation\2020_08_01", "_".join(os.path.basename(ortho).split("_")[2:5])+"_cleaned.shp")
     crownmap2020_tag.to_file(dir_out2)
     print("finish cleaning")
        #run crown avoidance code
     crown_avoided=crown_avoid(dir_out2)
     dir_out3=dir_out2.replace("cleaned.shp","_avoided.shp")
     crown_avoided.to_file(dir_out3)
     print("finish crown avoidance")

        #we merge the other fields from the original shapefile on global id
     orig= gpd.read_file(crown2020_path)
     improved= gpd.read_file(dir_out3)
     orig=orig[['Mnemonic', 'Latin','CrownCondi', 'Illuminati', 'Lianas',  'Inclinatio', 
            'Notes','stem_X', 'stem_Y',  'centroid_X',
            'centroid_Y', 'stemDist','DBH', 'crownArea', 'GlobalID',
            'Editor', 'EditDate','Person', 'FieldDate','Creator']]
     final=improved.merge(orig, on='GlobalID', how='left')
     final.to_file(dir_out3.replace("avoided.shp","improved.shp"))
     print("finish merging fields")
     crown2020_path=dir_out3.replace("avoided.shp","improved.shp")




target_orthomosaic=[file for file in orthomosaic_list if '2020_06_15' in file]




#crown predict the next previous date
tile_folder=r"D:\BCI_50ha_timeseries_local_alignment\tiles\2020_06_15"
if not os.path.exists(tile_folder):
    os.makedirs(tile_folder)

tile_ortho(target_orthomosaic[0],100,20,tile_folder)


dir_out= os.path.join(r"D:\crown_maps\crown_segmentation\2020_06_15")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
crown_segment(tile_folder,crown2020_path,dir_out)
print("finish instance segmentation")


crownmap=gpd.read_file(dir_out)
crownmap2020_tag = crownmap.sort_values('score', ascending=False)
crownmap2020_tag = crownmap.drop_duplicates('GlobalID', keep='first')
dir_out2= os.path.join(r"D:\crown_maps\crown_segmentation\2020_06_15", "2020_06_15_cleaned.shp")
crownmap2020_tag.to_file(dir_out2)

#run crown avoidance code
crown_avoided=crown_avoid(dir_out2)
dir_out3=dir_out2.replace("cleaned.shp","_avoided.shp")
crown_avoided.to_file(dir_out3)

#we merge the other fields from the original shapefile on global id
orig= gpd.read_file(crown2020_path)
improved= gpd.read_file(dir_out3)
orig=orig[['Mnemonic', 'Latin','CrownCondi', 'Illuminati', 'Lianas',  'Inclinatio', 
       'Notes','stem_X', 'stem_Y',  'centroid_X',
       'centroid_Y', 'stemDist','DBH', 'crownArea', 'GlobalID',
       'Editor', 'EditDate','Person', 'FieldDate','Creator']]
final=improved.merge(orig, on='GlobalID', how='left')
final.to_file(dir_out3.replace("avoided.shp","improved.shp"))















#read in the local alignment orthomosaic

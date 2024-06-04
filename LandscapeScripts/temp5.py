#basics 
import matplotlib.pyplot as plt
import os
import shutil
import rasterio
import numpy as np
import  cv2
from arosics import COREG, COREG_LOCAL
from LandscapeScripts.raster_tools import crop_raster
from LandscapeScripts.raster_tools import combine_ortho_dsm
import geopandas as gpd
from shapely.geometry import box as box1
import pandas as pd
from segment_anything import SamPredictor, sam_model_registry
import torch
from shapely.geometry import Polygon, MultiPolygon, box, shape
from shapely.ops import transform

wd_path= r"D:\BCI_50ha"
ortho_path = os.path.join(wd_path, "Orthophoto")
dsm_path = os.path.join(wd_path, "DSM")
product_path = os.path.join(wd_path, "Product")
cropped_path = os.path.join(wd_path, "Product_cropped")

BCI_50ha_shapefile = os.path.join(wd_path,"aux_files", "BCI_Plot_50ha.shp")
BCI_50ha = gpd.read_file(BCI_50ha_shapefile)
BCI_50ha.to_crs(epsg=32617, inplace=True)
BCI_50ha_buffer = box(BCI_50ha.bounds.minx-30, BCI_50ha.bounds.miny-30, BCI_50ha.bounds.maxx+30, BCI_50ha.bounds.maxy+30)  # Create a buffer around the plot

ortho_list = [file for file in os.listdir(ortho_path) if file.endswith(".tif")]
dsm_list= [file for file in os.listdir(dsm_path) if file.endswith(".tif")]  

#combine the orthomosaics and the dsm
for ortho, dsm in zip(ortho_list, dsm_list):
    ortho_file = os.path.join(ortho_path, ortho)
    dsm_file = os.path.join(dsm_path, dsm)
    out_file = os.path.join(wd_path, "Product", ortho)
    if not os.path.exists(out_file):
        combine_ortho_dsm(ortho_file, dsm_file, out_file)
    else:
        print(f"Skipping {ortho} because it already exists")

#crop the orthomosaics
product_list = [file for file in os.listdir(product_path) if file.endswith(".tif")]
#crop the orthomosaic
for product in product_list:
    if not os.path.exists(os.path.join(cropped_path, product)):
        crop_raster(os.path.join(product_path, product), os.path.join(cropped_path, product), BCI_50ha_buffer)
    else:
        print(f"Skipping {product} because it already exists")


#we need to align them all horizontally
reference1= os.path.join(cropped_path, ortho_list[69])
successful_alignments = [file for file in os.listdir(os.path.join(wd_path, "Product_global")) if file.endswith(".tif")]
for orthomosaic in ortho_list[68::-1]:
    print(orthomosaic)
    if orthomosaic != ortho_list[69]:
        target = os.path.join(cropped_path, orthomosaic)
        
        global_path = target.replace("orthomosaic.tif","aligned_global.tif").replace("Product_cropped","Product_global")
        if not os.path.exists(global_path):
            kwargs2 = { 'path_out': global_path,
                    'fmt_out': 'GTIFF',
                    'r_b4match': 2,
                    's_b4match': 2,
                    'max_shift': 200,
                    'max_iter': 20,
                    'align_grids':True,
                    'match_gsd': True,
                    'binary_ws': False
                }
            alignment_successful = False
            while not alignment_successful and successful_alignments:
                try:
                    CR= COREG(reference1, target, **kwargs2,ws=(2048,2048))
                    CR.calculate_spatial_shifts()
                    CR.correct_shifts()
                    print("Global alignment successful")
                    successful_alignments.append(global_path) # Add successful alignment to the list
                    alignment_successful = True
                except:
                    print("Global alignment failed, retrying with the previous successful alignment")
                    reference1 = os.path.join(wd_path,"Product_global",successful_alignments.pop()) # Use the last successful alignment as reference

reference1= os.path.join(cropped_path, ortho_list[69])
successful_alignments = [reference1]
for orthomosaic in ortho_list[70:]:
    print(orthomosaic)
    if orthomosaic != ortho_list[69]:
        target = os.path.join(cropped_path, orthomosaic)
        
        global_path = target.replace("orthomosaic.tif","aligned_global.tif").replace("Product_cropped","Product_global")
        if not os.path.exists(global_path):
            kwargs2 = { 'path_out': global_path,
                    'fmt_out': 'GTIFF',
                    'r_b4match': 2,
                    's_b4match': 2,
                    'max_shift': 200,
                    'max_iter': 20,
                    'align_grids':True,
                    'match_gsd': True,
                    'binary_ws': False
                }
            alignment_successful = False
            while not alignment_successful and successful_alignments:
                try:
                    CR= COREG(reference1, target, **kwargs2,ws=(2048,2048))
                    CR.calculate_spatial_shifts()
                    CR.correct_shifts()
                    print("Global alignment successful")
                    successful_alignments.append(global_path) # Add successful alignment to the list
                    alignment_successful = True
                except:
                    print("Global alignment failed, retrying with the previous successful alignment")
                    reference1 = os.path.join(wd_path,"Product_global",successful_alignments.pop()) # Use the last successful alignment as reference

shutil.copy(os.path.join(cropped_path, ortho_list[69]),os.path.join(wd_path,"Product_global", ortho_list[69]).replace("orthomosaic.tif","aligned_global.tif"))


def generate_grid(sub, tile_size):
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
        return gridInfo
def tile_ortho(sub, buffer, output_folder, gridInfo):
    with rasterio.open(sub) as src:
        for idx, row in gridInfo.iterrows():
            geom2 = box(row['xmin']-buffer, row['ymin']-buffer, row['xmax']+buffer, row['ymax']+buffer)
            out_image, out_transform = rasterio.mask.mask(src, [geom2], crop=True)
            # Update metadata for the output raster
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            base_name = os.path.basename(sub)
            output_filename = f"{base_name.replace('aligned_global.tif', 'tile')}_{idx}.tif"
            filename=os.path.join(output_folder,output_filename)
            with rasterio.open(filename, "w", **out_meta) as dest:
                dest.write(out_image)

tile_folder_base = os.path.join(wd_path, "Product_tiles")
global_list= [file for file in os.listdir(os.path.join(wd_path, "Product_global")) if file.endswith(".tif")]

# Generate the grid once
grid = generate_grid(os.path.join(wd_path,"Product_global", global_list[0]), 100)

for num in range(0,len(global_list)):
    print(global_list[num])
    tile_folder1=os.path.join(tile_folder_base, f"{global_list[num].replace('aligned_global.tif','')}")
    if not os.path.exists(tile_folder1):
        os.makedirs(tile_folder1)
    if len(os.listdir(tile_folder1)) == 50:
        print(f"Skipping {tile_folder1} because it already contains 49 files")
        continue
    print(tile_folder1)
    tile_ortho(os.path.join(wd_path,"Product_global", global_list[num]),30,tile_folder1, grid)


folder_out= os.path.join(wd_path, "tiles_local")
os.makedirs(folder_out, exist_ok=True)

list_folder = [folder for folder in os.listdir(tile_folder_base) if os.path.isdir(os.path.join(tile_folder_base, folder))]
for index in range(0,50):
    print(f"Processing index {index}")
    #loop backwards
    reference= os.path.join(tile_folder_base, list_folder[69], f"{list_folder[69]}tile_{index}.tif")
    for i in range(68, -1, -1):
        print(f"Processing {list_folder[i]}")
        target= os.path.join(tile_folder_base, list_folder[i], f"{list_folder[i]}tile_{index}.tif")
        output_path = os.path.join(folder_out, list_folder[i])
        os.makedirs(output_path, exist_ok=True)
        output_path2 = os.path.join(output_path, f"{list_folder[i]}tile_{index}.tif")
        kwargs = {
            'grid_res': 200,
            'window_size': (512, 512),
            'path_out': output_path2,
            'fmt_out': 'GTIFF',
            'q': False,
            'min_reliability': 30,
            'r_b4match': 2,
            's_b4match': 2,
            'max_shift': 100,
            'nodata': (0, 0),
            'match_gsd': True,
            'align_grids': True
        }
        try:
            if os.path.isfile(output_path2):
                print(f"File {output_path2} already exists")
                continue
            CR = COREG_LOCAL(reference, target, **kwargs)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            reference = output_path2
        except Exception as e:
            print(f"Error: {e}. Trying to align target to the previous reference.")
        
    #loop forwards
    reference= os.path.join(tile_folder_base, list_folder[69], f"{list_folder[69]}tile_{index}.tif")
    for i in range(70, len(list_folder)):
        print(f"Processing {list_folder[i]}")
        
        target= os.path.join(tile_folder_base, list_folder[i], f"{list_folder[i]}tile_{index}.tif")
        output_path = os.path.join(folder_out, list_folder[i])
        os.makedirs(output_path, exist_ok=True)
        output_path2 = os.path.join(output_path, f"{list_folder[i]}tile_{index}.tif")
        kwargs = {
            'grid_res': 200,
            'window_size': (512, 512),
            'path_out': output_path2,
            'fmt_out': 'GTIFF',
            'q': False,
            'min_reliability': 30,
            'r_b4match': 2,
            's_b4match': 2,
            'max_shift': 200,
            'nodata': (0, 0),
            'match_gsd': True,
            'align_grids': True
        }
        try:
            if os.path.isfile(output_path2):
                print(f"File {output_path2} already exists")
                continue
            CR = COREG_LOCAL(reference, target, **kwargs)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            reference = output_path2
        except Exception as e:
            print(f"Error: {e}. Trying to align target to the previous reference.")


#segmentation portion
#preprocessing the crown map 2022 manually
#open the improved version
improved_2022= r"D:\BCI_50ha\crownmap_datapub\BCI_50ha_2022_09_29_crownmap_improved\BCI_50ha_2022_09_29_crownmap_improved.shp"
improved_2022_gdf= gpd.read_file(improved_2022)
#define the dst folder 
dst_folder=r"D:\BCI_50ha\crownmap\BCI_50ha_2022_09_29_crownmap"
for idx in range(1,50):
    print(idx)
    improved_2022_gdf.to_file(os.path.join(dst_folder,f"crownmap_tile_{idx}.shp"))

#lets fucking segment
MODEL_TYPE = "vit_h"
checkpoint = r"C:\Users\VasquezV\repo\crown-segment\models\sam_vit_h_4b8939.pth"
device = 'cuda'
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
#sam.to(device=device)  #requires cuda cores
mask_predictor = SamPredictor(sam)


tile_local=r"D:\BCI_50ha\tiles_local"
crownmap_out_folder=r"D:\BCI_50ha\segmented_crownmap_2022"
#segment tile timeseries
def crown_segment2(tile_path,shp,output_shapefile):
        with rasterio.open(tile_path) as src:
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
        print(len(all_polygons),"crowns segmented")
        final_gdf = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=src.crs)
        final_gdf.to_file(output_shapefile)

#sequential segmentation
indexes=[2]
for index in indexes:
    shp=os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2022_09_29_crownmap",f"crownmap_tile_{index}.shp")
    list_folder =os.listdir(tile_local)
    for folder in list_folder[48::-1]:
        print(folder)
        path_desieredindex= os.path.join(tile_local, folder, f"{folder}tile_{index}.tif")
        output_path= os.path.join(crownmap_out_folder, f"{folder}tile_{index}.shp")
        crown_segment2(tile_path=path_desieredindex,shp=shp,output_shapefile=output_path)
        shp=output_path
    for folder in list_folder[50:]:
        print(folder)
        path_desieredindex= os.path.join(tile_local, folder, f"{folder}tile_{index}.tif")
        output_path= os.path.join(crownmap_out_folder, f"{folder}tile_{index}.shp")
        crown_segment2(tile_path=path_desieredindex,shp=shp,output_shapefile=output_path)
        shp=output_path


for index in indexes:
    shp=os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2022_09_29_crownmap",f"crownmap_tile_{index}.shp")
    shpprj= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2022_09_29_crownmap",f"crownmap_tile_{index}.prj")
    shpcpg= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2022_09_29_crownmap",f"crownmap_tile_{index}.cpg")
    shpshx= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2022_09_29_crownmap",f"crownmap_tile_{index}.shx")
    shpdbj= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2022_09_29_crownmap",f"crownmap_tile_{index}.dbf")
    shutil.copy(shp, os.path.join(crownmap_out_folder, f"BCI_50ha_2022_09_29_tile_{index}.shp"))
    shutil.copy(shpprj, os.path.join(crownmap_out_folder, f"BCI_50ha_2022_09_29_tile_{index}.prj"))
    shutil.copy(shpcpg, os.path.join(crownmap_out_folder, f"BCI_50ha_2022_09_29_tile_{index}.cpg"))
    shutil.copy(shpshx, os.path.join(crownmap_out_folder, f"BCI_50ha_2022_09_29_tile_{index}.shx"))
    shutil.copy(shpdbj, os.path.join(crownmap_out_folder, f"BCI_50ha_2022_09_29_tile_{index}.dbf"))

file= gpd.read_file(os.path.join(crownmap_out_folder, f"BCI_50ha_2022_09_29_tile_{index}.shp"))
file= file[['area', 'score', 'tag', 'GlobalID', 'geometry']]
file.to_file(os.path.join(crownmap_out_folder, f"BCI_50ha_2022_09_29_tile_{index}.shp"))




#create gdf of that tile and that crown
indexes=[2]
for index in indexes:
    print(f"Processing index {index}")
    crown_dates= [filename for filename in os.listdir(crownmap_out_folder) if filename.endswith(f'tile_{index}.shp')]
    all_gdf=[]
    for date in range(0,len(crown_dates)):
        path= os.path.join(crownmap_out_folder, crown_dates[date])
        gdf= gpd.read_file(path)
        gdf = gdf.to_crs("EPSG:32617")  # Set a common CRS
        date = '_'.join(crown_dates[date].split('_')[2:5])
        tile= os.path.join(tile_local, f"BCI_50ha_{date}_", f"BCI_50ha_{date}_tile_{index}.tif")
        gdf['date']=date
        gdf['tile']=tile
        gdf= gdf[['area', 'score', 'date','tile','tag', 'GlobalID', 'geometry']]
        all_gdf.append(gdf)

    all= pd.concat(all_gdf, ignore_index=True)
    all['tag'] = all['tag'].astype(str)
    reference_gdf= gpd.read_file(r"D:\BCI_50ha\crownmap_datapub\BCI_50ha_2022_09_29_crownmap_raw\BCI_50ha_2022_2023_crownmap.shp")
    reference_gdf["tag"] = reference_gdf["tag"].astype(str)
    
    merged_gdf = all.merge(reference_gdf[["tag","latin"]], on='tag', how='left')

    merged_gdf['outlier'] = merged_gdf.groupby('tag')['area'].transform(lambda x: np.abs(x - x.mean()) > 2 * x.std())
    merged_gdf.to_file(f"D:\BCI_50ha\crownmap\crownmap_{index}_2022.shp")
    
    merged_gdf= gpd.read_file(f"D:\BCI_50ha\crownmap\crownmap_{index}_2022.shp")
    merged_gdf = merged_gdf.dropna(subset=['latin'])
    unique_tags = merged_gdf['tag'].unique()
    folder_out_crown= os.path.join(wd_path, "segmented_crownmap_2022", f"tile_{index}")
    os.makedirs(folder_out_crown, exist_ok=True)
    for tag in unique_tags:
        thistree = merged_gdf[merged_gdf['tag'] == str(tag)]
        subset2 = thistree[thistree['date'] == '2022_09_29']
        subset2_geom = subset2.iloc[0]['geometry']
        main_box = box1(subset2.total_bounds[0] - 5, subset2.total_bounds[1] - 5, subset2.total_bounds[2] + 5, subset2.total_bounds[3] + 5)
        for i in range(0, len(thistree)):
            crown_sp1 = thistree.iloc[i]['latin'].split(" ")[0]
            crown_sp2 = thistree.iloc[i]['latin'].split(" ")[1]
            tag = thistree.iloc[i]['tag']
            date = thistree.iloc[i]['date']
            outlier = thistree.iloc[i]['outlier']  # Check for outlier
            folder_out_crown= os.path.join(wd_path, "segmented_crownmap_2022", f"tile_{index}")
            if not os.path.exists(folder_out_crown):
                os.makedirs(folder_out_crown)
            aux_folder=output_path = os.path.join(folder_out_crown,f"{crown_sp1}_{crown_sp2}_{tag}")
            if not os.path.exists(aux_folder):
                os.makedirs(aux_folder)
            output_path = os.path.join(folder_out_crown,f"{crown_sp1}_{crown_sp2}_{tag}",f"{crown_sp1}_{crown_sp2}_{tag}_{date}.png")
            geom = thistree.iloc[i]['geometry']
            
            # Choose which geometry to use based on the outlier flag
            if outlier==True:
                geom = subset2_geom
                print(f"Outlier detected for crown {tag} in date {date}. Using the geometry from 2020-08-01.")
            
            
            with rasterio.open(thistree.iloc[i]['tile']) as src:
                out_image, out_transform = rasterio.mask.mask(src, [main_box], crop=True)
                out_meta = src.meta.copy()
                x_min, y_min = out_transform * (0, 0)
                xres, yres = out_transform[0], out_transform[4]
                transformed_geom = transform(lambda x, y: ((x - x_min) / xres, (y - y_min) / yres), geom)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(out_image.transpose((1, 2, 0))[:, :, 0:3])
                ax.plot(*transformed_geom.exterior.xy, color='red')
                for interior in transformed_geom.interiors:
                    ax.plot(*interior.xy, color='red')
                ax.axis('off')
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)




tile0=r"D:\BCI_50ha\segmented_crownmap_2022\tile_0"
tile1=r"D:\BCI_50ha\segmented_crownmap_2022\tile_1"

#which folders in tile0 are in tile1
folders_tile0= [folder for folder in os.listdir(tile0) if os.path.isdir(os.path.join(tile0, folder))]
folders_tile1= [folder for folder in os.listdir(tile1) if os.path.isdir(os.path.join(tile1, folder))]

for folder in folders_tile0:
    if folder in folders_tile1:
        print(folder)

Astronium_graveolens_7663
Guatteria_dumetorum_7666
Pouteria_reticulata_7655
Quararibea_asterolepis_7957
Quararibea_asterolepis_7960
Virola_surinamensis_7665
Zanthoxylum_ekmanii_603076

cordia_bicolor_7650
platypodium_elegans_7652

#crown segmentation
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
import uuid

#functions
#tile ortho divides and orthomosaic into tiles by taking an orthomosaic the x and y tile size and adding a buffer, the tile size its adjusted 
#to deal with the residual of the division
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
#segmentation function takes a folder with tiles and a shapefile with the polygons to be segmented, it identifies the polygons within the tile
#then it segments by inputting a torch tensor with multiple boxes, the output is a geodataframe with the polygons and the score of the segmentation
#out of the 3 masks it selects the higuest score for the box and then it transforms the mask back to utm coordinates
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
        crowns=shp
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
def crown_avoid(dir):
    crown_avoidance = gpd.read_file(dir)
    crown_avoidance['geometry'] = crown_avoidance.geometry.buffer(0)
    sindex = crown_avoidance.sindex

    modifications = {}  # Dictionary to collect modifications

    for idx, polygon in crown_avoidance.iterrows():
        possible_matches_index = list(sindex.intersection(polygon['geometry'].bounds))
        possible_matches = crown_avoidance.iloc[possible_matches_index]
        adjacents = possible_matches[possible_matches.geometry.intersects(polygon['geometry']) & (possible_matches.index != idx)]
        if adjacents.empty:
            continue
        else:
            print("found adjacents", adjacents["tag"])
            for adj_idx, adj_polygon in adjacents.iterrows():
                if polygon.geometry.area > adj_polygon.geometry.area:
                    # Collect modification for polygon at idx
                    modifications[idx] = modifications.get(idx, polygon.geometry).difference(adj_polygon.geometry)
                elif polygon.geometry.area < adj_polygon.geometry.area:
                    # Collect modification for adjacent polygon
                    modifications[adj_idx] = modifications.get(adj_idx, adj_polygon.geometry).difference(polygon.geometry)

    # Apply collected modifications
    for idx, new_geom in modifications.items():
        crown_avoidance.at[idx, 'geometry'] = new_geom

    return crown_avoidance

MODEL_TYPE = "vit_h"
checkpoint = r"/home/vasquezv/BCI_50ha/aux_files/sam_vit_h_4b8939.pth"
device = 'cuda'
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
#sam.to(device=device)  #requires cuda cores
mask_predictor = SamPredictor(sam)

#Working directory
wd_path= r"/home/vasquezv/BCI_50ha"
BCI_2022_raw= os.path.join(wd_path,"crownmap/BCI_50ha_2022_2023_crownmap_raw.shp")

#open the shapefile
crownmap2022=gpd.read_file(BCI_2022_raw)
crownmap2022["GlobalID"] = [uuid.uuid4() for _ in range(len(crownmap2022))]
crownmap2022["GlobalID"] = crownmap2022["GlobalID"].astype(str)

orthomosaics=os.listdir(os.path.join(wd_path,"Product_local2"))
tile_folder= os.path.join(wd_path,"tiles")
os.makedirs(tile_folder, exist_ok=True)



#The first 3 dates are important since they should be compared to the reference and and not to the avoided crown map
#2022_09_29
#segmentation
tile_ortho(os.path.join(wd_path,"Product_local2","BCI_50ha_2022_09_29_aligned_local2.tif"),100,30,tile_folder)
crown_segment(tile_folder,crownmap2022,os.path.join(wd_path,"crownmap/BCI_50ha_2022_09_29_crownmap_segmented.shp"))                                                    
crownmap_improved=gpd.read_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_09_29_crownmap_segmented.shp"))

#filter repeated crowns
for index, crown in crownmap_improved.iterrows():
    crown_original = crownmap2022[crownmap2022["GlobalID"] == crown["GlobalID"]].iloc[0]
    intersection = crown.geometry.intersection(crown_original.geometry)
    union = crown.geometry.union(crown_original.geometry)
    iou = intersection.area / union.area if union.area > 0 else 0
    crownmap_improved.loc[index, "iou"] = iou

crownmap2022_improved = crownmap_improved.sort_values("iou", ascending=False).drop_duplicates("GlobalID", keep="first")
crownmap2022_improved.to_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_09_29_crownmap_segmented.shp"))

#crown avoidance
crownmap_avoidance = crown_avoid(os.path.join(wd_path,"crownmap/BCI_50ha_2022_09_29_crownmap_segmented.shp"))
crownmap_avoidance.to_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_09_29_crownmap_avoid.shp"))

#2022_10_27
tile_ortho(os.path.join(wd_path,"Product_local2","BCI_50ha_2022_10_27_aligned_local2.tif"),100,30,tile_folder)
crown_segment(tile_folder,crownmap2022,os.path.join(wd_path,"crownmap/BCI_50ha_2022_10_27_crownmap_segmented.shp"))
crownmap_improved=gpd.read_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_10_27_crownmap_segmented.shp"))
for index, crown in crownmap_improved.iterrows():
    crown_original = crownmap2022[crownmap2022["GlobalID"] == crown["GlobalID"]].iloc[0]
    intersection = crown.geometry.intersection(crown_original.geometry)
    union = crown.geometry.union(crown_original.geometry)
    iou = intersection.area / union.area if union.area > 0 else 0
    crownmap_improved.loc[index, "iou"] = iou
crownmap2022_improved = crownmap_improved.sort_values("iou", ascending=False).drop_duplicates("GlobalID", keep="first")
crownmap2022_improved.to_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_10_27_crownmap_segmented.shp"))
crownmap_avoidance = crown_avoid(os.path.join(wd_path,"crownmap/BCI_50ha_2022_10_27_crownmap_segmented.shp"))
crownmap_avoidance.to_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_10_27_crownmap_avoid.shp"))

#2022_08_24
tile_ortho(os.path.join(wd_path,"Product_local2","BCI_50ha_2022_08_24_aligned_local2.tif"),100,30,tile_folder)
crown_segment(tile_folder,crownmap2022,os.path.join(wd_path,"crownmap/BCI_50ha_2022_08_24_crownmap_segmented.shp"))
crownmap_improved=gpd.read_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_08_24_crownmap_segmented.shp"))
for index, crown in crownmap_improved.iterrows():
    crown_original = crownmap2022[crownmap2022["GlobalID"] == crown["GlobalID"]].iloc[0]
    intersection = crown.geometry.intersection(crown_original.geometry)
    union = crown.geometry.union(crown_original.geometry)
    iou = intersection.area / union.area if union.area > 0 else 0
    crownmap_improved.loc[index, "iou"] = iou
crownmap2022_improved = crownmap_improved.sort_values("iou", ascending=False).drop_duplicates("GlobalID", keep="first")
crownmap2022_improved.to_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_08_24_crownmap_segmented.shp"))
crownmap_avoidance = crown_avoid(os.path.join(wd_path,"crownmap/BCI_50ha_2022_08_24_crownmap_segmented.shp"))
crownmap_avoidance.to_file(os.path.join(wd_path,"crownmap/BCI_50ha_2022_08_24_crownmap_avoid.shp"))

#Get the names of the orthomosaic files
dates= os.listdir(os.path.join(wd_path,"Product_local2"))
dates= [date for date in dates if date.endswith(".tif")]
info_ortho=pd.DataFrame(columns=["filename","ortho_path"])
info_ortho["filename"]=dates
info_ortho["ortho_path"]=info_ortho["filename"].apply(lambda x: os.path.join(wd_path,"Product_local2",x))
info_ortho['date'] = info_ortho['filename'].apply(lambda x: "_".join(x.split("_")[2:5]))

#reference files
date_reference= info_ortho.loc[48].values[2]
crownmap_reference= info_ortho.loc[48].values[1].replace("Product_local2","crownmap").replace("_aligned_local2.tif","_crownmap_avoid.shp")

for i in range(47, -1, -1):
    ortho=info_ortho.loc[i].values[1]
    date=info_ortho.loc[i].values[2]
    crownmap_out= crownmap_reference.replace(date_reference,date).replace("crownmap_avoid.shp","crownmap_segmented.shp")
    tile_ortho(ortho,100,30,tile_folder)
    crown_segment(tile_folder,crownmap_reference,crownmap_out)
    crownmap_improved=gpd.read_file(crownmap_out)
    crownmap_prv=gpd.read_file(crownmap_reference)
    for index, crown in crownmap_improved.iterrows():
        crown_original = crownmap_prv[crownmap_prv["GlobalID"] == crown["GlobalID"]].iloc[0]
        intersection = crown.geometry.intersection(crown_original.geometry)
        union = crown.geometry.union(crown_original.geometry)
        iou = intersection.area / union.area if union.area > 0 else 0
        crownmap_improved.loc[index, "iou"] = iou
    crownmap_improved = crownmap_improved.sort_values("iou", ascending=False).drop_duplicates("GlobalID", keep="first")
    crownmap_improved.to_file(crownmap_out)
    crownmap_avoidance = crown_avoid(crownmap_out)
    crownmap_avoidance.to_file(crownmap_out.replace("_crownmap_segmented.shp","_crownmap_avoid.shp"))
    crownmap_reference = crownmap_out
    date_reference = date

date_reference= info_ortho.loc[50].values[2]
crownmap_reference= info_ortho.loc[50].values[1].replace("Product_local2","crownmap").replace("_aligned_local2.tif","_crownmap_avoid.shp")

for i in range(51,106,1):
    ortho=info_ortho.loc[i].values[1]
    date=info_ortho.loc[i].values[2]
    crownmap_out= crownmap_reference.replace(date_reference,date).replace("crownmap_avoid.shp","crownmap_segmented.shp")
    tile_ortho(ortho,100,30,tile_folder)
    crown_segment(tile_folder,crownmap_reference,crownmap_out)
    crownmap_improved=gpd.read_file(crownmap_out)
    crownmap_prv=gpd.read_file(crownmap_reference)
    for index, crown in crownmap_improved.iterrows():
        crown_original = crownmap_prv[crownmap_prv["GlobalID"] == crown["GlobalID"]].iloc[0]
        intersection = crown.geometry.intersection(crown_original.geometry)
        union = crown.geometry.union(crown_original.geometry)
        iou = intersection.area / union.area if union.area > 0 else 0
        crownmap_improved.loc[index, "iou"] = iou
    crownmap_improved = crownmap_improved.sort_values("iou", ascending=False).drop_duplicates("GlobalID", keep="first")
    crownmap_improved.to_file(crownmap_out)
    crownmap_avoidance = crown_avoid(crownmap_out)
    crownmap_avoidance.to_file(crownmap_out.replace("_crownmap_segmented.shp","_crownmap_avoid.shp"))
    crownmap_reference = crownmap_out
    date_reference = date



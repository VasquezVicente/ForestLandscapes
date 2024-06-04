#raster tools
import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from rasterio.mask import mask
import os
import pandas as pd
import cv2
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import box as box1
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
#tile ortho function
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
        return gridInfo
#combine function, remeber to deal with uint8 problem which wont hold a value above 255
def combine_ortho_dsm(ortho_path,dsm_path, output_path):
    with rasterio.open(ortho_path) as src:
        ortho_data = src.read()
        ortho_meta = src.meta.copy()
    with rasterio.open(dsm_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
        dem_data=np.where(dem_data==dem_meta['nodata'],0,dem_data)
    resampled_dem = np.zeros((ortho_meta['height'], ortho_meta['width']), dtype=ortho_data.dtype)
    reproject(
    dem_data, resampled_dem,
    src_transform=dem_meta['transform'],
    src_crs=dem_meta['crs'],
    dst_transform=ortho_meta['transform'],
    dst_crs=ortho_meta['crs'],
    resampling=Resampling.nearest)
    ortho_data[3,:,:] = resampled_dem
    ortho_meta['count'] = 4
    with rasterio.open(output_path, 'w', **ortho_meta) as dst:
        dst.write(ortho_data)
#crop function 
def crop_raster(input_path, output_path, shapely_polygon):
    with rasterio.open(input_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [shapely_polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
#reproject function
def calc_chm(dtm_path, ortho_path, output_path):
    with rasterio.open(dtm_path) as src:
        dtm_data = src.read(1)
        dtm_meta = src.meta.copy()
    with rasterio.open(ortho_path) as src:
        ortho_data = src.read()
        ortho_meta = src.meta.copy()

    reprojected_dtm = np.zeros((ortho_meta['height'], ortho_meta['width']), dtype=dtm_data.dtype)
    reproject(
        dtm_data, reprojected_dtm,
        src_transform=dtm_meta['transform'],
        src_crs=dtm_meta['crs'],
        dst_transform=ortho_meta['transform'],
        dst_crs=ortho_meta['crs'],
        resampling=Resampling.nearest)
    chm=ortho_data[3,:,:]-reprojected_dtm
    chm=np.where(chm>150,0,chm)
    chm=np.where(chm<0,0,chm)
    ortho_data[3,:,:]=chm
    ortho_meta['count'] = 4
    with rasterio.open(output_path, 'w', **ortho_meta) as dst:
        dst.write(ortho_data)
#still not ready
def align_vertically(tar,ref,output_path,re_ref=True, max_shift=50, tar_is_ortho=True):
    with rasterio.open(tar) as src:
            tar_data=src.read(4)
            tar_meta=src.meta.copy()
    with rasterio.open(ref) as src:
        ref_data=src.read(1)
        ref_meta=src.meta.copy()
    if re_ref==True:
        reprojected_ref = np.zeros((tar_meta['height'], tar_meta['width']), dtype=np.float64)
        reproject(
            ref_data, reprojected_ref,
            src_transform=ref_meta['transform'],
            src_crs=ref_meta['crs'],
            dst_transform=tar_meta['transform'],
            dst_crs=tar_meta['crs'],
            resampling=Resampling.nearest)
    delta = tar_data - reprojected_ref
    delta_nodata= delta[~np.isnan(delta)]
    median_key=np.median(delta_nodata)
    print("median_key",median_key)

    with rasterio.open(tar) as src:
        tar_data = src.read()  
        tar_meta = src.meta.copy()
    # Subtract median_key from elements of ortho_data that are greater than zero
        tar_data[3,:,:] = np.where((tar_data[3,:,:]>0)|(tar_data[3,:,:]<255), tar_data[3,:,:] - np.int8(median_key), 0)
        plt.imshow(tar_data[3,:,:])
        plt.show()
    


    plt.hist(delta[~np.isnan(delta)], bins=100)	
    plt.show()
    np.median(delta[~np.isnan(delta)])
    delta_notna=delta_shift[delta_shift != 0]
    np.median(delta_shift[~np.isnan(delta_shift)])


    delta_notna=delta_shift[~np.isnan(delta_shift)]
    median_key=np.median(delta_notna)
    print("median_key",median_key)   
    with rasterio.open(tar) as src:
        tar_data = src.read()  
        tar_meta = src.meta.copy()
    # Subtract median_key from elements of ortho_data that are greater than zero
    tar_data[3,:,:] = np.where(tar_data[3,:,:] > 0, tar_data[3,:,:] - np.int8(median_key), tar_data[3,:,:])
    tar_meta['count'] = 4
    with rasterio.open(output_path, 'w', **tar_meta) as dst:
        dst.write(tar_data)

    return median_key

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

#this function takes a shapefile with polygons and removes the overlapping polygons, it also removes the multipolygons.
#the functions tries to terminate the while loop earlier when there is no more overlap but sometimes it does not terminate and for that the counter
#to end the loop after 10 iterations
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


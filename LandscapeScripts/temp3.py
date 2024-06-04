import os
import time
import copy
import shutil
import numpy as np
import pandas as pd
from shapely.ops import transform
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from skimage.transform import rescale
from shapely.geometry import Polygon, MultiPolygon, box, shape
from matplotlib.patches import Rectangle
from segment_anything import SamPredictor, sam_model_registry
import torch
from rasterio.features import rasterize, geometry_mask, shapes
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
from rasterio import windows
from rasterio.plot import show
i
from rasterio.merge import merge
from datetime import datetime
from arosics import COREG, COREG_LOCAL
from arosics import COREG_LOCAL
from shapely.geometry import box as box1
from matplotlib import pyplot as plt
import numpy as np
import json
import ndjson
import cv2
import shapely
#!/usr/bin/env python
import os
import yaml
import click
import shutil
from glob import glob
from ffmpeg import FFmpeg
from datetime import datetime
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory

BCI_50ha_directory = os.getcwd()
print(BCI_50ha_directory)

#BCI_50ha_directory = r"D:\BCI_50ha"
#functions
def combine_ortho_dsm(ortho_path,dsm_path, output_path):
    with rasterio.open(ortho_path) as src:
        ortho_data = src.read()
        ortho_meta = src.meta.copy()
    with rasterio.open(dsm_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
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
            base_name = os.path.basename(sub)
            output_filename = f"{base_name.replace('orthomosaic.tif', 'tile')}_{idx}.tif"
            filename=os.path.join(output_folder,output_filename)
            with rasterio.open(filename, "w", **out_meta) as dest:
                dest.write(out_image)
    return gridInfo
def transform_geometry(geom, transform):
    x_min, y_min = transform * (0, 0)
    xres, yres = transform[0], transform[4]
    return shapely.ops.transform(lambda x, y: ((x - x_min) / xres, (y - y_min) / yres), geom)
def parse_info(filename):
    parts = os.path.splitext(filename)[0].split('_')
    return {
        'genus': parts[0],
        'species': parts[1],
        'crown_id': int(parts[2]),
        'date': datetime(*map(int, parts[-3:])),
    }
def reformat_image(inputfile, config, outputdir):
    video_size = config['video_size']
    date_fmt = config.get('date_format', None)
    species_fmt = config.get('species_format', None)
    padding = config['padding']
    left, right, top, bottom = (
        padding.get(side, 0)
        for side in ('left', 'right', 'top', 'bottom')
    )

    base = os.path.basename(inputfile)
    info = parse_info(base)
    outputfile = os.path.join(outputdir, base)

    total_size = (
        video_size[0] + left + right,
        video_size[1] + top + bottom
    )

    out = Image.new('RGB', total_size)

    # Draw original image
    with Image.open(inputfile) as im:
        w, h = im.size
        xoff = ((video_size[0] - w) // 2) + left
        yoff = ((video_size[1] - h) // 2) + top
        out.paste(im, (xoff, yoff))

    if date_fmt is not None:
        datestr = info['date'].strftime(date_fmt['format'])
        draw = ImageDraw.Draw(out)
        draw.text(
            date_fmt['xy'],
            datestr,
            **date_fmt.get('draw_kwargs', {})
        )

    if species_fmt is not None:
        specstr = species_fmt['format'].format(**info)
        draw = ImageDraw.Draw(out)
        draw.text(
            species_fmt['xy'],
            specstr,
            **species_fmt.get('draw_kwargs', {})
        )

    out.save(outputfile)
    return outputfile

wd_path= r"/home/vasquezv/BCI_50ha"
wd_path= r"D:\BCI_50ha"
#50ha shapefile for boundaries
#read the 50ha shape file and transform it to UTM 17N
BCI_50ha_shapefile = os.path.join(wd_path,"aux_files", "BCI_Plot_50ha.shp")
BCI_50ha = gpd.read_file(BCI_50ha_shapefile)
BCI_50ha.to_crs(epsg=32617, inplace=True)
BCI_50ha_buffer = box(BCI_50ha.bounds.minx-20, BCI_50ha.bounds.miny-20, BCI_50ha.bounds.maxx+20, BCI_50ha.bounds.maxy+20)  # Create a buffer around the plot

#working directory

path_orthomosaic = os.path.join(wd_path, "Orthophoto")
path_DSM = os.path.join(wd_path, "DSM")
path_output= os.path.join(wd_path, "Product")
path_cropped= os.path.join(wd_path, "Product_cropped")
tile_folder_base= os.path.join(wd_path, "tiles")
base_output_path = os.path.join(wd_path, "output")    
crownmap_out_folder=os.path.join(wd_path, "crownmap_out")
if not os.path.exists(path_orthomosaic):   
    os.makedirs(path_orthomosaic)
if not os.path.exists(path_DSM):
    os.makedirs(path_DSM)
if not os.path.exists(path_output):
    os.makedirs(path_output)
if not os.path.exists(path_cropped):
    os.makedirs(path_cropped)
if not os.path.exists(tile_folder_base):
    os.makedirs(tile_folder_base)
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)
if not os.path.exists(crownmap_out_folder):
    os.makedirs(crownmap_out_folder)

#COMBINE DSM AND ORTHOPHOTO
orthomosaics = [filename for filename in os.listdir(path_orthomosaic) if filename.endswith('.tif')]
DSMs= [filename for filename in os.listdir(path_DSM) if filename.endswith('.tif')]
DSMs_replaced = [filename.replace('dsm', 'orthomosaic') for filename in DSMs]

if DSMs_replaced == orthomosaics:
    print("The lists correspond to each other.")
else:
    print("The lists do not correspond to each other.")

for i in range(0, len(orthomosaics)):
    ortho= os.path.join(path_orthomosaic, orthomosaics[i])
    DSM= os.path.join(path_DSM, DSMs[i])
    if not os.path.exists(os.path.join(path_cropped, orthomosaics[i])):
        combine_ortho_dsm(ortho, DSM, os.path.join(path_output, orthomosaics[i]))
        crop_raster(os.path.join(path_output, orthomosaics[i]), os.path.join(path_cropped, orthomosaics[i]), BCI_50ha_buffer)
        print(f"Combined {orthomosaics[i]} and {DSMs[i]}")
    else:
        print(f"File {orthomosaics[i]} already exists")


#after cropping we need to globally coregister all orthomosaics to increment precision of the local coregistration
#here the cropped BCI_50ha_2023_05_23 is the reference orthomosaic for the global coregistration
folder_global= os.path.join(wd_path, "global_coreg")
if not os.path.exists(folder_global):
    os.makedirs(folder_global)
targets= [filename for filename in os.listdir(path_cropped) if filename.endswith('.tif')]
reference_orthomosaic=r"D:\BCI_50ha\Product_cropped\BCI_50ha_2023_05_23_orthomosaic.tif"
print("starting the global correction of the first orthomosaic")



for tar in targets:
    output_path2=os.path.join(folder_global,tar.replace("orthomosaic.tif","aligned_global.tif"))
    print(output_path2)
    target= os.path.join(path_cropped, tar)
    try:
            kwargs2 = {
                'path_out': output_path2,
                'fmt_out': 'GTIFF',
                'r_b4match': 2,
                's_b4match': 2,
                'max_shift': 200,
                'max_iter': 20,
                'align_grids':True,
                'match_gsd': False,
            }
            CR = COREG(reference_orthomosaic, target, **kwargs2)
            CR.calculate_spatial_shifts()
            CR.correct_shifts()
            continue  # Exit the loop if no RuntimeError
    except RuntimeError as e:
            print(f"Error processing {target}: {e}")
            continue  # Go to the next iteration if RuntimeError



#tile the first 20 orthomosaics
orthomosaics= [filename for filename in os.listdir(path_cropped) if filename.endswith('.tif')]

for num in range(0,len(orthomosaics)):
    print(orthomosaics[num])
    tile_folder1=os.path.join(tile_folder_base, f"{orthomosaics[num].replace('.tif','')}")
    if not os.path.exists(tile_folder1):
        os.makedirs(tile_folder1)
    if len(os.listdir(tile_folder1)) == 50:
        print(f"Skipping {tile_folder1} because it already contains 49 files")
        continue
    print(os.path.join(path_cropped, orthomosaics[num]))
    print(tile_folder1)
    tile_ortho(os.path.join(path_cropped, orthomosaics[num]),100,20,tile_folder1)


# Define the list of folders
list_folder = [folder for folder in os.listdir(tile_folder_base) if os.path.isdir(os.path.join(tile_folder_base, folder))]
for index in range(0,50):
    print(f"Processing index {index}")
    til1 = None
    prev_til1 = None
    for i in range(1, len(list_folder)):
        if til1 is not None:
            prev_til1 = til1
        if til1 is None:
            til1 = os.path.join(tile_folder_base, list_folder[i-1], f"{list_folder[i-1].replace('_orthomosaic','_tile')}_{index}.tif")
        else:
            til1 = os.path.join(base_output_path, list_folder[i-1],  f"{list_folder[i-1].replace('_orthomosaic','_tile')}_{index}.tif")
        til2 = os.path.join(tile_folder_base, list_folder[i],  f"{list_folder[i].replace('_orthomosaic','_tile')}_{index}.tif")
        output_path = os.path.join(base_output_path, list_folder[i])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path2 = os.path.join(output_path, f"{list_folder[i].replace('_orthomosaic','_tile')}_{index}.tif")
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
            'match_gsd': False
        }
        try:
            if os.path.isfile(output_path2):
                print(f"File {output_path2} already exists")
                continue
            CRL = COREG_LOCAL(til1, til2, **kwargs)
            CRL.calculate_spatial_shifts()
            CRL.correct_shifts()
        except Exception as e:
            print(f"Error: {e}. Trying to align til2 to the previous til1.")
            if prev_til1 is not None:
                try:
                    if os.path.isfile(output_path2):
                        print(f"File {output_path2} already exists")
                        continue
                    CRL = COREG_LOCAL(prev_til1, til2, **kwargs)
                    CRL.calculate_spatial_shifts()
                    CRL.correct_shifts()
                except Exception as e:
                    print(f"Error: {e}. Failed to align til2 to the previous til1.")


#lets fucking segment
MODEL_TYPE = "vit_h"
checkpoint = r"C:\Users\VasquezV\repo\crown-segment\models\sam_vit_h_4b8939.pth"
device = 'cuda'
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
#sam.to(device=device)  #requires cuda cores
mask_predictor = SamPredictor(sam)


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

indexes=[16]
for index in indexes:
    shp=os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2020_08_01_crownmap",f"tile_{index}_crownmap.shp")
    list_folder = [folder for folder in os.listdir(base_output_path) if os.path.isdir(os.path.join(base_output_path, folder))]
    for folder in list_folder:
        path_desieredindex= os.path.join(base_output_path, folder, f"{folder.replace('_orthomosaic','_tile')}_{index}.tif")
        output_path= os.path.join(crownmap_out_folder, f"{folder.replace('_orthomosaic','_tile')}_{index}.shp")
        crown_segment2(tile_path=path_desieredindex,shp=shp,output_shapefile=output_path)

indexes=[16]
for index in indexes:
    shp=os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2020_08_01_crownmap",f"tile_{index}_crownmap.shp")
    list_folder = [folder for folder in os.listdir(base_output_path) if os.path.isdir(os.path.join(base_output_path, folder))]
    for folder in list_folder[23::-1]:
        print(folder)
        path_desieredindex= os.path.join(base_output_path, folder, f"{folder.replace('_orthomosaic','_tile')}_{index}.tif")
        output_path= os.path.join(crownmap_out_folder, f"{folder.replace('_orthomosaic','_tile')}_{index}.shp")
        crown_segment2(tile_path=path_desieredindex,shp=shp,output_shapefile=output_path)
        shp=output_path
    for folder in list_folder[25:]:
        print(folder)
        path_desieredindex= os.path.join(base_output_path, folder, f"{folder.replace('_orthomosaic','_tile')}_{index}.tif")
        output_path= os.path.join(crownmap_out_folder, f"{folder.replace('_orthomosaic','_tile')}_{index}.shp")
        crown_segment2(tile_path=path_desieredindex,shp=shp,output_shapefile=output_path)
        shp=output_path

index=16
shp=os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2020_08_01_crownmap",f"tile_{index}_crownmap.shp")
shpprj= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2020_08_01_crownmap",f"tile_{index}_crownmap.prj")
shpcpg= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2020_08_01_crownmap",f"tile_{index}_crownmap.cpg")
shpshx= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2020_08_01_crownmap",f"tile_{index}_crownmap.shx")
shpdbj= os.path.join(r"D:\BCI_50ha\crownmap\BCI_50ha_2020_08_01_crownmap",f"tile_{index}_crownmap.dbf")

shutil.copy(shp, os.path.join(crownmap_out_folder, f"BCI_50ha_2020_08_01_tile_{index}.shp"))
shutil.copy(shpprj, os.path.join(crownmap_out_folder, f"BCI_50ha_2020_08_01_tile_{index}.prj"))
shutil.copy(shpcpg, os.path.join(crownmap_out_folder, f"BCI_50ha_2020_08_01_tile_{index}.cpg"))
shutil.copy(shpshx, os.path.join(crownmap_out_folder, f"BCI_50ha_2020_08_01_tile_{index}.shx"))
shutil.copy(shpdbj, os.path.join(crownmap_out_folder, f"BCI_50ha_2020_08_01_tile_{index}.dbf"))

file= gpd.read_file(os.path.join(crownmap_out_folder, f"BCI_50ha_2020_08_01_tile_{index}.shp"))
file= file[['area', 'score', 'tag', 'GlobalID', 'geometry']]
file.to_file(os.path.join(crownmap_out_folder, f"BCI_50ha_2020_08_01_tile_{index}.shp"))

#create gdf of that tile and that crown
#list all shps that end with tile_18.shp
indexes=[16]
for index in indexes:
    print(f"Processing index {index}")
    segmented_crowns_18= [filename for filename in os.listdir(crownmap_out_folder) if filename.endswith(f'tile_{index}.shp')]
    all_gdf=[]
    for date_18 in range(0,len(segmented_crowns_18)):
        path= os.path.join(crownmap_out_folder, segmented_crowns_18[date_18])
        gdf= gpd.read_file(path)
        gdf = gdf.to_crs("EPSG:32617")  # Set a common CRS
        date = '_'.join(segmented_crowns_18[date_18].split('_')[2:5])
        tile= os.path.join(base_output_path, f"BCI_50ha_{date}_orthomosaic", f"BCI_50ha_{date}_tile_{index}.tif")
        gdf['date']=date
        gdf['tile']=tile
        gdf= gdf[['area', 'score', 'date','tile','tag', 'GlobalID', 'geometry']]
        all_gdf.append(gdf)

    all= pd.concat(all_gdf, ignore_index=True)
    all['tag'] = all['tag'].astype(str)

    reference_gdf= gpd.read_file(r"D:\BCI_50ha\crownmap_datapub\BCI_50ha_2020_08_01_crownmap_raw\Crowns_2020_08_01_MergedWithPlotData.shp")
    reference_gdf=reference_gdf.rename(columns={"Tag":"tag"})
    reference_gdf["tag"] = reference_gdf["tag"].astype(str)
    
    merged_gdf = all.merge(reference_gdf[["tag","Latin"]], on='tag', how='left')



    merged_gdf['outlier'] = merged_gdf.groupby('tag')['area'].transform(lambda x: np.abs(x - x.mean()) > 2 * x.std())
    merged_gdf.to_file(f"D:\BCI_50ha\crownmap\crownmap_{index}.shp")

    unique_tags = merged_gdf['tag'].unique()

    for tag in unique_tags:
        thistree = merged_gdf[merged_gdf['tag'] == str(tag)]
        subset2 = thistree[thistree['date'] == '2020_08_01']
        subset2_geom = subset2.iloc[0]['geometry']
        main_box = box1(subset2.total_bounds[0] - 5, subset2.total_bounds[1] - 5, subset2.total_bounds[2] + 5, subset2.total_bounds[3] + 5)
        for i in range(0, len(thistree)):
            crown_sp1 = thistree.iloc[i]['Latin'].split(" ")[0]
            crown_sp2 = thistree.iloc[i]['Latin'].split(" ")[1]
            tag = thistree.iloc[i]['tag']
            date = thistree.iloc[i]['date']
            outlier = thistree.iloc[i]['outlier']  # Check for outlier
            folder_out_crown= os.path.join(wd_path, "crown_segmentation", f"tile_{index}")
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





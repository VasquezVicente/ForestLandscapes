#main.py
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
import rasterio
from shapely import box
from full_workflow.utils import crop_raster
from segment_anything import SamPredictor
from segment_anything import sam_model_registry

#hyperparameters
buffer=20 #20 meters to the shape to avoid issues

#path to plot shape
plots_panama=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Drone_Pilot_Data\aux_files\Panama_forest_plots.shp"
plots_panama=gpd.read_file(plots_panama)
sherman_shape= plots_panama[plots_panama['Plot']=='sherman'].reset_index()
sherman_shape= box(sherman_shape.total_bounds[0]-20,sherman_shape.total_bounds[1]-20,sherman_shape.total_bounds[2]+20,sherman_shape.total_bounds[3]+20)

#paths to reference and target orthomosaics
sherman_2021=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2021\SANLORENZO_75ha_2021_05_18_P4P\Orthophoto\SANLORENZO_75ha_2021_05_18_orthomosaic.tif"
sherman_2025=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2025\SANLORENZO_sherman_2025_01_07_M3M\20250107_shermancrane_m3e_rgb.tif"
sherman_2021_out=os.path.join(os.path.dirname(sherman_2021),os.path.basename(sherman_2021).replace("_orthomosaic.tif","_shermansub.tif"))
sherman_2025_out=os.path.join(os.path.dirname(sherman_2025),os.path.basename(sherman_2025).replace(".tif","_shermansub.tif"))

crop_raster(sherman_2021, sherman_2021_out,sherman_shape)
crop_raster(sherman_2025, sherman_2025_out,sherman_shape)

#align the target to the reference
sherman_2021_out_align=os.path.join(os.path.dirname(sherman_2021),os.path.basename(sherman_2021_out).replace("_shermansub.tif","_shermanaligned.tif"))

align_orthomosaics(sherman_2025_out,sherman_2021_out,sherman_2021_out_align) # if this fails, the orthomosaic needs to be georeferenced in arcgis pro

#next step is to transfer the labels, for that we need crown segment
# I will test the 50ha plot first

path_orthomosaic=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2025\BCI_50ha_2025_03_10_M3E\Orthophoto\BCI_50ha_2025_03_10_orthomosaic.tif"
path_cropped= path_orthomosaic.replace('_orthomosaic.tif', '_cropped.tif')
path_crownmap=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\geodataframes\BCI_50ha_2022_2023_crownmap_raw.shp"

#cropping to only include 50ha plot and not the ava and 10ha plot
crownmap= gpd.read_file(path_crownmap)
bounds= crownmap.total_bounds
shapebci= box(bounds[0]-20,bounds[1]-20,bounds[2]+20,bounds[3]+20)
crop_raster(path_orthomosaic,path_cropped,shapebci)

#AI model loading
MODEL_TYPE = "vit_h"
checkpoint = r"D:\BCI_50ha\aux_files\sam_vit_h_4b8939.pth"
device = 'cuda'
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
sam.to(device=device)  #requires cuda cores, add # if you dont have cuda installed or access to a GPU
mask_predictor = SamPredictor(sam)

#3 functions, process crown data does the last two together with default 100 and 30. or do both individually.
from full_workflow.utils import process_crown_data, tile_ortho,crown_segment

wd= r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\crownmap2025" # working directory
tile_folder=os.path.join(wd,"tiles") #tiles folder 
os.makedirs(tile_folder, exist_ok=True) #create tiles folder
orthomosaic= path_cropped  # path cropped is the same than orthomosaic to be used for segmentation
crownmap2025= path_crownmap.replace("BCI_50ha_2022_2023_crownmap_raw.shp","BCI_50ha_2025_crownmap_raw.shp") #output path

reference=gpd.read_file(path_crownmap) # read the reference crown map
tile_ortho(orthomosaic,75,20,tile_folder)# tile the orthomosaic to be used for segmentation
crown_segment(tile_folder,reference,mask_predictor,crownmap2025)









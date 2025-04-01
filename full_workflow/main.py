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
bci_shape= plots_panama[plots_panama['Plot']=='50 Ha Plot'].reset_index()
bci_shape= box(bci_shape.total_bounds[0]-20,bci_shape.total_bounds[1]-20,bci_shape.total_bounds[2]+20,bci_shape.total_bounds[3]+20)

#paths to reference and target orthomosaics
path_orthomosaic=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2024\BCI_50ha_2024_11_12_M3E\Orthophoto\BCI_50ha_2024_11_12_orthomosaic.tif"
path_crownmap=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\geodataframes\BCI_50ha_2022_2023_crownmap_raw.shp"
path_cropped= path_orthomosaic.replace('_orthomosaic.tif','_cropped.tif')
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
tile_ortho(orthomosaic,80,20,tile_folder)# tile the orthomosaic to be used for segmentation
crown_segment(tile_folder,reference,mask_predictor,crownmap2025)


reference.columns
crownmap_improved = gpd.read_file(crownmap2025)
crownmap_improved.columns
    
for index, crown in crownmap_improved.iterrows():
        crown_original = reference[reference["GlobalID"] == crown["GlobalID"]].iloc[0]
        intersection = crown.geometry.intersection(crown_original.geometry)
        union = crown.geometry.union(crown_original.geometry)
        iou = intersection.area / union.area if union.area > 0 else 0
        crownmap_improved.loc[index, "iou"] = iou
crownmap_filtered = crownmap_improved.sort_values("iou", ascending=False).drop_duplicates("GlobalID", keep="first")
crown_merged= crownmap_filtered.merge(reference[['GlobalID','latin','mnemonic']],left_on='GlobalID', right_on='GlobalID', how='right')
crown_merged.to_file(r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\geodataframes\BCI_50ha_2025_crownmap_final.shp")














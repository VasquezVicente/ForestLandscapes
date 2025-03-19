#main.py
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
import rasterio
from shapely import box
from full_workflow.utils import crop_raster, align_orthomosaics

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





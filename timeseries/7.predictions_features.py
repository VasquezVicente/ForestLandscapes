import os
import pandas as pd
import geopandas as gpd
import rasterio
from shapely import box
import matplotlib.pyplot as plt
import shapely
import numpy as np
from skimage.filters.rank import entropy
import pickle
from rasterio.mask import mask
import matplotlib.patches as patches
from shapely.affinity import affine_transform
from matplotlib.backends.backend_pdf import PdfPages
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from skimage.morphology import disk
import sys
import warnings
from utils import calculate_glcm_features


#load polygons
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")


species_subset= crowns[crowns['latin']=='Prioria copaifera'].reset_index()

pixel_unmixing=gpd.read_file(os.path.join(data_path,'aux_files/pixel_unmixing.shp'))
gv_pixels = []  # For GV (Green Vegetation)
npv_pixels = []  # For NPV (Non-photosynthetic Vegetation)
shadow_pixels =[]  # for shadows

for i, (_, row) in enumerate(pixel_unmixing.iterrows()):
    path_orthomosaic = os.path.join(data_path,'orthomosaic_aligned_local',row['filename'])
    with rasterio.open(path_orthomosaic) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True)
        red = out_image[0]  # Band 1 (Red)
        green = out_image[1]  # Band 2 (Green)
        blue = out_image[2]  # Band 3 (Blue)
        red= np.where(red==0, np.nan, red)
        green= np.where(green==0, np.nan, green)
        blue= np.where(blue==0, np.nan, blue)
        if row['endpoint']== 'pv':
            gv_pixels.append(np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1))
        elif row['endpoint']== 'npv':
            npv_pixels.append(np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1))
        elif row['endpoint']== 'shadow':
            shadow_pixels.append(np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1))


gv_pixels = np.vstack(gv_pixels)
gv_pixels_clean = gv_pixels[~np.isnan(gv_pixels)]
gv_endmember = np.nanmean(gv_pixels, axis=0)

npv_pixels = np.vstack(npv_pixels)
npv_pixels_clean = npv_pixels[~np.isnan(npv_pixels)]
npv_endmember = np.nanmean(npv_pixels, axis=0)

shadow_pixels = np.vstack(shadow_pixels)
shadow_pixels_clean = shadow_pixels[~np.isnan(shadow_pixels)]
shadow_endmember = np.nanmean(shadow_pixels, axis=0)


#stack the endmembers
A = np.vstack([gv_endmember, npv_endmember,shadow_endmember]).T 
A_inv = np.linalg.pinv(A)
A_inv.shape


#angles for covariance matrix
window_size = 5  # 5x5 window
angles = [0, 45, 90, 135]  # Azimuths in degrees


species_subset[species_subset['date'].isna()]
#extract the features, crown based
counter=0
list_ortho = [f for f in os.listdir(os.path.join(data_path, 'orthomosaic_aligned_local')) 
              if f.endswith('.tif') and not f.endswith('.tif.ovr')]
for orthomosaic in list_ortho:
    print(f"Processing orthomosaic: ", orthomosaic)
    date = "_".join(orthomosaic.split("_")[2:5])
    with rasterio.open(os.path.join(data_path,'orthomosaic_aligned_local',orthomosaic)) as src:
        polygons_date= species_subset[species_subset['date']==date]
        for idx, row in polygons_date.iterrows():
            counter += 1
            print(f"  Processed rows: {counter}", end='\r')  # keep it clean in the terminal
            out_image, out_transform = mask(src, [row.geometry], crop=True)
            red = out_image[0]  # Band 1 (Red)
            green = out_image[1]  # Band 2 (Green)
            blue = out_image[2]  # Band 3 (Blue)
            elev= out_image[3]
            red= np.where(red==0, np.nan, red)
            green= np.where(green==0,np.nan, green)
            blue=np.where(blue==0,np.nan,blue)
            elev= np.where(elev==0, np.nan, elev)

            #gray covariance matrix analysis
            gray_img = np.mean(out_image.transpose(1, 2, 0), axis=-1)
            
            gray_img = np.uint8((gray_img - np.nanmin(gray_img)) / (np.nanmax(gray_img) - np.nanmin(gray_img)) * 255)
            gcor_values = calculate_glcm_features(gray_img, window_size=window_size, angles=angles)

            #this is for nvp and gv
            image_data = np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1)
            surface_fractions = np.dot(image_data, A_inv.T)
            gv_fraction = surface_fractions[:, 0].reshape(out_image.shape[1], out_image.shape[2])
            npv_fraction = surface_fractions[:, 1].reshape(out_image.shape[1], out_image.shape[2])
            shadow_fraction= surface_fractions[:, 2].reshape(out_image.shape[1], out_image.shape[2])

            # Calculate RCC: Red / (Red + Green + Blue)
            rcc = red / (green + red + blue)
            gcc= green/ (green + red + blue)
            bcc= blue / (green + red + blue)
            ExG= (2*green)- (red+blue)

            #lets see entropy, I belive this term will improve the algorithm 
            structuring_element = disk(5)
            entropy_image = entropy(gray_img, structuring_element)


            rccM= np.nanmean(rcc)
            gccM=np.nanmean(gcc)
            bccM=np.nanmean(bcc)
            ExGM= np.nanmean(ExG)
            gvM=np.nanmean(gv_fraction)
            npvM=np.nanmean(npv_fraction)
            shadowM=np.nanmean(shadow_fraction)
            rSD = np.nanstd(red)
            gSD= np.nanstd(green)
            bSD= np.nanstd(blue)
            ExGSD= np.nanstd(ExG)
            gvSD=np.nanstd(gv_fraction)
            npvSD= np.nanstd(npv_fraction)
            gcorSD= np.nanstd(gcor_values)
            gcorMD= np.nanmedian(gcor_values)
            entropyM=np.nanmean(entropy_image)
            elevSD= np.nanstd(elev)

            species_subset.at[idx, 'rccM'] = rccM
            species_subset.at[idx, 'gccM'] = gccM
            species_subset.at[idx, 'bccM'] = bccM
            species_subset.at[idx, 'ExGM'] = ExGM
            species_subset.at[idx, 'gvM'] = gvM
            species_subset.at[idx, 'npvM'] = npvM
            species_subset.at[idx, 'shadowM'] = shadowM
            species_subset.at[idx, 'rSD'] = rSD
            species_subset.at[idx, 'gSD'] = gSD
            species_subset.at[idx, 'bSD'] = bSD
            species_subset.at[idx, 'ExGSD'] = ExGSD
            species_subset.at[idx, 'gvSD'] = gvSD
            species_subset.at[idx, 'npvSD'] = npvSD
            species_subset.at[idx, 'gcorSD'] = gcorSD
            species_subset.at[idx, 'gcorMD'] = gcorMD
            species_subset.at[idx, 'entropy']= entropyM
            species_subset.at[idx, 'elevSD']= elevSD


species_subset=species_subset.drop(columns=['geometry'])
species_subset.to_csv(r"timeseries/dataset_predictions/prioria_sgbt.csv")

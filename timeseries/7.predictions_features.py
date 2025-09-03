import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import shapely
from skimage.filters.rank import entropy
from shapely.affinity import affine_transform
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk
from timeseries.utils import calculate_glcm_features
from timeseries.utils import create_consensus_polygon
from timeseries.utils import create_overlap_density_map
from timeseries.utils import generate_leafing_pdf
import matplotlib.pyplot as plt

#load polygons

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"                 ## path to the data folder
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")                         ## orthomosaics locally aligned location 
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")  ## location of the timeseries of polygons
crowns=gpd.read_file(path_crowns)                                                      ## read the file using geopandas
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")       ## polygon ID defines the identity of tree plus date it was taken
species_subset= crowns[crowns['latin']=='Chrysophyllum cainito'].reset_index()         ## geodataframe to be used as template to extract features

## adding analysis of shape and size of the crown
A_inv=np.array([[ 0.00174702,  0.01227676, -0.01372143],   #green vegetation endpoint 
                [ 0.02120641, -0.02059761,  0.00372091],   #non green vegetation endpoint 
                [-0.08542822,  0.06046723,  0.02937261]])  #shadow vegetation endpoint 

#angles for covariance matrix
window_size = 5  # 5x5 window
angles = [0, 45, 90, 135]  # Azimuths in degrees

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
species_subset.to_csv(r"timeseries/dataset_predictions/cainito_sgbt.csv")

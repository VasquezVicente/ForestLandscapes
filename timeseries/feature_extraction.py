import os
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio
from rasterio.mask import mask
import os
import shapely
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patches as patches
from shapely.affinity import affine_transform
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage import img_as_ubyte
from timeseries.timeseries_tools import generate_leafing_pdf

#PATHS
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
#data_path=r"C:\Users\Vicente\Documents\Data"
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
labels_path=r"timeseries/export_2025_03_05.csv"
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")

#list of orthomosaics
orthomosaic_list=os.listdir(orthomosaic_path)

#open gdf containing polygons
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")
#open df containing labels
labels=pd.read_csv(labels_path)
#merge labels and crowns, keeping only labeled ones
crowns_labeled= labels.merge(crowns[['area', 'score', 'tag', 'iou', 'geometry','polygon_id']],
                              left_on="polygon_id",
                                right_on="polygon_id",
                                  how="left")


crowns_labeled= crowns_labeled[crowns_labeled["segmentation"]=="good"]
crowns_labeled= crowns_labeled[crowns_labeled['isFlowering']=="no"]
crowns_labeled.columns

crowns_labeled_avg = crowns_labeled.groupby("polygon_id").agg({
    "leafing": "mean",  
    "latin": "first",  
    "geometry": "first",
    "date":"first",
    "area":"first",
    "tag":"first",
    "iou":"first",
    "score":"first"
}).reset_index()


crowns_labeled_avg = gpd.GeoDataFrame(crowns_labeled_avg, geometry='geometry')
crowns_labeled_avg.set_crs("EPSG:32617", allow_override=True, inplace=True)  


crowns_labeled_0 = crowns_labeled_avg[crowns_labeled_avg['leafing'] == 0]
generate_leafing_pdf(r"plots/leafing0.pdf",crowns_labeled_0,orthomosaic_path,crowns_per_page=12)
crowns_labeled_0_csv= crowns_labeled_0.drop(columns='geometry')
crowns_labeled_0_csv.to_csv(r"timeseries/check_0.csv")


crowns_labeled_01 = crowns_labeled_avg[(crowns_labeled_avg['leafing'] > 0) & (crowns_labeled_avg['leafing'] < 100)]
generate_leafing_pdf(r"plots/leafing01.pdf",crowns_labeled_01,orthomosaic_path,crowns_per_page=12)
crowns_labeled_01_csv= crowns_labeled_01.drop(columns='geometry')
crowns_labeled_01_csv.to_csv(r"timeseries/check_01.csv")


crowns_labeled_1= crowns_labeled_avg[crowns_labeled_avg['leafing'] == 100]

crowns_labeled_0_csv= pd.read_csv(r"timeseries/check_0.csv")
crowns_labeled_0_updated= crowns_labeled_0.merge(crowns_labeled_0_csv[["polygon_id","leafing"]], left_on="polygon_id", right_on="polygon_id")
crowns_labeled_0_updated['leafing']= crowns_labeled_0_updated['leafing_y']

crowns_labeled_0_csv= pd.read_csv(r"timeseries/check_01.csv")
crowns_labeled_01_updated= crowns_labeled_01.merge(crowns_labeled_01_csv[["polygon_id","leafing"]], left_on="polygon_id", right_on="polygon_id")
crowns_labeled_01_updated['leafing']= crowns_labeled_01_updated['leafing_y']


crowns_labeled_all = pd.concat([crowns_labeled_01_updated, crowns_labeled_0_updated, crowns_labeled_1])
crowns_labeled_all = gpd.GeoDataFrame(crowns_labeled_all, geometry='geometry')
crowns_labeled_all.set_crs("EPSG:32617", allow_override=True, inplace=True)  

crowns_labeled_all=crowns_labeled_all[['polygon_id','latin', 'geometry', 'date', 'area', 'tag',
       'iou', 'score', 'leafing']]


gv_pixels = []  # For GV (Green Vegetation)
npv_pixels = []  # For NPV (Non-photosynthetic Vegetation)

for i, (_, row) in enumerate(crowns_labeled_1.iterrows()):
    if i >= 2:  # Stop after 10 images
        break
    path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")
    with rasterio.open(path_orthomosaic) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True)
        plt.imshow(out_image.transpose(1,2,0))
        plt.show()
        red = out_image[0]  # Band 1 (Red)
        green = out_image[1]  # Band 2 (Green)
        blue = out_image[2]  # Band 3 (Blue)
        red= np.where(red==0, np.nan, red)
        green= np.where(green==0,np.nan, green)
        blue=np.where(blue==0,np.nan,blue)
        gv_pixels.append(np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1))

import seaborn as sns
gv_pixels = np.vstack(gv_pixels)
gv_pixels_clean = gv_pixels[~np.isnan(gv_pixels)]
plt.figure(figsize=(10, 6))
sns.histplot(gv_pixels_clean, label="GV Pixels", color="green", kde=True, alpha=0.5)
plt.show()
gv_endmember = np.nanmean(gv_pixels, axis=0)

for i, (_, row) in enumerate(crowns_labeled_0.iterrows()):
    if i >= 2:  
        break
    path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")
    with rasterio.open(path_orthomosaic) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True) 
        red = out_image[0]  
        green = out_image[1] 
        blue = out_image[2]
        red= np.where(red>230,red,0)
        green= np.where(green>230,green,0)
        blue=np.where(blue>230,blue,0)  
        plt.imshow(np.dstack((red,green,blue)))
        plt.show()
        red= np.where(red==0, np.nan, red)
        green= np.where(green==0,np.nan, green)
        blue=np.where(blue==0,np.nan,blue)
        npv_pixels.append(np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1))


npv_pixels = np.vstack(npv_pixels)
npv_endmember = np.nanmean(npv_pixels, axis=0)

#stack the endmembers
A = np.vstack([gv_endmember, npv_endmember]).T 
A_inv = np.linalg.pinv(A)
A_inv.shape

plt.figure(figsize=(8, 6))
bands = ['Red', 'Green', 'Blue']
plt.plot(bands, gv_endmember, label='GV Endmember (Green Vegetation)', marker='o')
plt.plot(bands, npv_endmember, label='NPV Endmember (Non-Photosynthetic Vegetation)', marker='o')

plt.xlabel('Spectral Bands')
plt.ylabel('Reflectance Value')
plt.title('Spectral Response of Endmembers')
plt.legend()
plt.grid(True)
plt.show()


#anges for covariance matrix
window_size = 5  # 5x5 window
angles = [0, 45, 90, 135]  # Azimuths in degrees

def calculate_glcm_features(image, window_size=5, angles=[0, 45, 90, 135]):
    glcm_features = []
    for angle in angles:
        # Calculate the GLCM for the specified angle
        glcm = graycomatrix(img_as_ubyte(image), distances=[1], angles=[np.deg2rad(angle)], symmetric=True, normed=True) 
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        glcm_features.append(correlation)
    
    return glcm_features


#extract the features, crown based 
for i, (_, row) in enumerate(crowns_labeled_all.iterrows()):
    print(f"Processing iteration {i + 1} of {len(crowns_labeled_all)}")
    path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")
    with rasterio.open(path_orthomosaic) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True)
        red = out_image[0]  # Band 1 (Red)
        green = out_image[1]  # Band 2 (Green)
        blue = out_image[2]  # Band 3 (Blue)
        red= np.where(red==0, np.nan, red)
        green= np.where(green==0,np.nan, green)
        blue=np.where(blue==0,np.nan,blue)

        #gray covariance matrix analysis
        gray_img = np.mean(out_image.transpose(1, 2, 0), axis=-1)
        gray_img = np.uint8((gray_img - np.nanmin(gray_img)) / (np.nanmax(gray_img) - np.nanmin(gray_img)) * 255)
        gcor_values = calculate_glcm_features(gray_img, window_size=window_size, angles=angles)

        #this is for nvp and gv
        image_data = np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1)
        surface_fractions = np.dot(image_data, A_inv.T)
        gv_fraction = surface_fractions[:, 0].reshape(out_image.shape[1], out_image.shape[2])
        npv_fraction = surface_fractions[:, 1].reshape(out_image.shape[1], out_image.shape[2])

        # Calculate RCC: Red / (Red + Green + Blue)
        rcc = red / (green + red + blue)
        gcc= green/ (green + red + blue)
        bcc= blue / (green + red + blue)
        ExG= (2*green)- (red+blue)

        rccM= np.nanmean(rcc)
        gccM=np.nanmean(gcc)
        bccM=np.nanmean(bcc)
        ExGM= np.nanmean(ExG)
        gvM=np.nanmean(gv_fraction)
        npvM=np.nanmean(npv_fraction)
        rSD = np.nanstd(red)
        gSD= np.nanstd(green)
        bSD= np.nanstd(blue)
        ExGSD= np.nanstd(ExG)
        gvSD=np.nanstd(gv_fraction)
        npvSD= np.nanstd(npv_fraction)
        gcorSD= np.nanstd(gcor_values)
        gcorMD= np.nanmedian(gcor_values)


        crowns_labeled_avg.at[i, 'rccM'] = rccM
        crowns_labeled_avg.at[i, 'gccM'] = gccM
        crowns_labeled_avg.at[i, 'bccM'] = bccM
        crowns_labeled_avg.at[i, 'ExGM'] = ExGM
        crowns_labeled_avg.at[i, 'gvM'] = gvM
        crowns_labeled_avg.at[i, 'npvM'] = npvM
        crowns_labeled_avg.at[i, 'rSD'] = rSD
        crowns_labeled_avg.at[i, 'gSD'] = gSD
        crowns_labeled_avg.at[i, 'bSD'] = bSD
        crowns_labeled_avg.at[i, 'ExGSD'] = ExGSD
        crowns_labeled_avg.at[i, 'gvSD'] = gvSD
        crowns_labeled_avg.at[i, 'npvSD'] = npvSD
        crowns_labeled_avg.at[i, 'gcorSD'] = gcorSD
        crowns_labeled_avg.at[i, 'gcorMD'] = gcorMD


crowns_labeled_avg.to_file(r"timeseries/training_dataset.shp")

        

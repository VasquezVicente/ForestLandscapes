import os
import pandas as pd
import geopandas as gpd
import rasterio
from shapely import box
import matplotlib.pyplot as plt
import shapely
from rasterio.mask import mask
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pickle
from rasterio.mask import mask
import shapely
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patches as patches
from shapely.affinity import affine_transform
from matplotlib.backends.backend_pdf import PdfPages
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import seaborn as sns
from skimage.filters.rank import entropy
from skimage.morphology import disk


#load polygons
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")


dipteryx_subset= crowns[crowns['latin']=='Dipteryx oleifera'].reset_index()
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

plt.figure(figsize=(8, 6))
bands = ['Red', 'Green', 'Blue']
plt.plot(bands, gv_endmember, label='GV Endmember (Green Vegetation)', marker='o')
plt.plot(bands, npv_endmember, label='NPV Endmember (Non-Photosynthetic Vegetation)', marker='o')
plt.plot(bands, shadow_endmember, label='Shadow Endmember (Shadow)', marker='o')

plt.xlabel('Spectral Bands')
plt.ylabel('Reflectance Value')
plt.title('Spectral Response of Endmembers')
plt.legend()
plt.grid(True)
plt.show()


#angles for covariance matrix
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
for i, (_, row) in enumerate(dipteryx_subset.iterrows()):
    print(f"Processing iteration {i + 1} of {len(dipteryx_subset)}")
    path_orthomosaic = os.path.join(data_path,'orthomosaic_aligned_local', f"BCI_50ha_{row['date']}_local.tif")
    with rasterio.open(path_orthomosaic) as src:
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

        dipteryx_subset.at[i, 'rccM'] = rccM
        dipteryx_subset.at[i, 'gccM'] = gccM
        dipteryx_subset.at[i, 'bccM'] = bccM
        dipteryx_subset.at[i, 'ExGM'] = ExGM
        dipteryx_subset.at[i, 'gvM'] = gvM
        dipteryx_subset.at[i, 'npvM'] = npvM
        dipteryx_subset.at[i, 'shadowM'] = shadowM
        dipteryx_subset.at[i, 'rSD'] = rSD
        dipteryx_subset.at[i, 'gSD'] = gSD
        dipteryx_subset.at[i, 'bSD'] = bSD
        dipteryx_subset.at[i, 'ExGSD'] = ExGSD
        dipteryx_subset.at[i, 'gvSD'] = gvSD
        dipteryx_subset.at[i, 'npvSD'] = npvSD
        dipteryx_subset.at[i, 'gcorSD'] = gcorSD
        dipteryx_subset.at[i, 'gcorMD'] = gcorMD
        dipteryx_subset.at[i, 'entropy']= entropyM
        dipteryx_subset.at[i, 'elevSD']= elevSD

dipteryx_subset=dipteryx_subset.drop(columns=['geometry'])

dipteryx_subset.to_csv(r"timeseries/dataset_predictions/dipteryx_sgbt.csv")

dipteryx_predict= pd.read_csv(r'timeseries/dataset_predictions/dipteryx_sgbt.csv')
dipteryx_predict.columns

with open(r'timeseries/models/xgb_model.pkl', 'rb') as file:
      model = pickle.load(file)

X=dipteryx_predict[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]

Y= dipteryx_predict[['area', 'score', 'tag', 'GlobalID', 'iou',
       'date', 'latin', 'polygon_id']]
X_predicted=model.predict(X)
df_final = Y.copy()  # Copy Y to keep the same structure
df_final['leafing_predicted'] = X_predicted

df_final['date'] = pd.to_datetime(df_final['date'], format='%Y_%m_%d')
df_final.columns

import matplotlib.pyplot as plt
import seaborn as sns

df_final = df_final.sort_values(by=['GlobalID', 'date'])

# Set plot size
plt.figure(figsize=(12, 6))

# Plot one line per GlobalID
sns.lineplot(data=df_final, x='date', y='leafing_predicted', hue='GlobalID', estimator=None, lw=2)

# Formatting
plt.xlabel('Date')
plt.ylabel('Leafing Predicted')
plt.title('Leafing Predictions Over Time for Each GlobalID')
plt.xticks(rotation=45)
plt.legend(title='GlobalID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.show()

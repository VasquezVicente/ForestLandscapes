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
import pickle



#load polygons
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"                 ## path to the data folder
crown_segmentation_check_path=os.path.join(data_path,'videos\cavallinesia_crown_check')
os.makedirs(crown_segmentation_check_path, exist_ok=True)
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")                         ## orthomosaics locally aligned location 
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")  ## location of the timeseries of polygons
crowns=gpd.read_file(path_crowns)                                                      ## read the file using geopandas
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")       ## polygon ID defines the identity of tree plus date it was taken
species_subset= crowns[crowns['latin']=='Dipteryx oleifera'].reset_index() 
####################################################################################################################
################correct the polygons before extracting the features##################################################
unique_globalids = species_subset['GlobalID'].unique()
all_gid_polygons = []
for gid in unique_globalids:
    subset = species_subset[species_subset['GlobalID'] == gid]
    grid_size = 0.1
    x_coords, y_coords, density_matrix = create_overlap_density_map(subset, grid_size=grid_size)
    consensus_polygon = create_consensus_polygon(x_coords, y_coords, density_matrix)

    #this now plots correctly aligned
    fig, ax = plt.subplots(figsize=(10, 10))
    extent = [subset.total_bounds[0], subset.total_bounds[2], subset.total_bounds[1], subset.total_bounds[3]]
    im = ax.imshow(density_matrix, extent=extent, origin='lower', cmap='hot', alpha=0.7)
    subset['geometry'].plot(ax=ax, edgecolor='white', facecolor='none', alpha=0.8, linewidth=1)
    #plot concensus polygon if it exists
    if consensus_polygon is not None:
        gpd.GeoSeries(consensus_polygon).plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2)
    plt.colorbar(im, ax=ax, label='Overlap Count')
    plt.title(f"Density Map with Polygons - {gid}")
    plt.show()

    # Calculate Hausdorff distances and replace high-distance polygons
    if consensus_polygon is not None:
        # First loop: calculate Hausdorff distances
        for idx, row in subset.iterrows():
            poly = row['geometry']
            distance = poly.hausdorff_distance(consensus_polygon)
            subset.loc[idx, 'hausdorff_distance'] = distance
            print(f"Polygon has Hausdorff distance: {distance:.2f}")

        # Calculate statistics for outlier detection
        distances = subset['hausdorff_distance']
        mean_distance = distances.mean()
        std_distance = distances.std()

        final_threshold= mean_distance + 1 * std_distance
        
        # Use the more conservative threshold

        print(f"Mean Hausdorff distance for {gid}: {mean_distance:.2f}")
        print(f"Standard deviation: {std_distance:.2f}")
        print(f"Z-score threshold (μ + 1.5σ): {final_threshold:.2f}")
        print(f"Final threshold used: {final_threshold:.2f}")
        
        # Second loop: replace only significant outliers
        replacements_made = 0
        for idx, row in subset.iterrows():
            if row['hausdorff_distance'] > final_threshold:
                print(f"Polygon with GlobalID {gid} has high Hausdorff distance: {row['hausdorff_distance']:.2f} - REPLACING")
                subset.loc[idx, 'geometry'] = consensus_polygon
                replacements_made += 1
        
        print(f"Replaced {replacements_made}/{len(subset)} polygons ({replacements_made/len(subset)*100:.1f}%)")

    #generate_leafing_pdf(subset, os.path.join(crown_segmentation_check_path, f"{gid}.pdf"), path_ortho, crowns_per_page=12, variables=['hausdorff_distance', 'date'])
    subset['area'] = subset['geometry'].area
    all_gid_polygons.append(subset)

final_gdf = gpd.GeoDataFrame(pd.concat(all_gid_polygons, ignore_index=True), crs=subset.crs)
final_gdf.columns


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
        polygons_date= final_gdf[final_gdf['date']==date]
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

            final_gdf.at[idx, 'rccM'] = rccM
            final_gdf.at[idx, 'gccM'] = gccM
            final_gdf.at[idx, 'bccM'] = bccM
            final_gdf.at[idx, 'ExGM'] = ExGM
            final_gdf.at[idx, 'gvM'] = gvM
            final_gdf.at[idx, 'npvM'] = npvM
            final_gdf.at[idx, 'shadowM'] = shadowM
            final_gdf.at[idx, 'rSD'] = rSD
            final_gdf.at[idx, 'gSD'] = gSD
            final_gdf.at[idx, 'bSD'] = bSD
            final_gdf.at[idx, 'ExGSD'] = ExGSD
            final_gdf.at[idx, 'gvSD'] = gvSD
            final_gdf.at[idx, 'npvSD'] = npvSD
            final_gdf.at[idx, 'gcorSD'] = gcorSD
            final_gdf.at[idx, 'gcorMD'] = gcorMD
            final_gdf.at[idx, 'entropy'] = entropyM
            final_gdf.at[idx, 'elevSD'] = elevSD


final_gdf=final_gdf.drop(columns=['geometry'])
final_gdf.to_csv(r"timeseries/dataset_predictions/dipteryx_sgbt.csv")


with open(r'timeseries/models/xgb_model.pkl', 'rb') as file:
      model = pickle.load(file)
#with open(r'timeseries/models/xgb_model_flower.pkl', 'rb') as file:
#      model_flower = pickle.load(file)

data=pd.read_csv(r"timeseries/dataset_predictions/dipteryx_sgbt.csv")


X=data[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',     #features for prediction
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]

Y= data[['area', 'score', 'tag', 'GlobalID', 'iou',                                        #identifiers
       'date', 'latin', 'polygon_id']]

X_predicted=model.predict(X)                                                      #predictions
#X_predict_flower= model_flower.predict(X)

df_final = Y.copy()  # Copy Y to keep the same structure
df_final['leafing_predicted'] = X_predicted
#df_final['isFlowering_predicted'] = X_predict_flower


#lets bring in the actual labels to clean it up
training_dataset=pd.read_csv(r"timeseries/dataset_training/train_sgbt.csv")
merged_final= df_final.merge(training_dataset[['polygon_id','leafing']], on='polygon_id', how='left')

#flowering correction
flower_correction=pd.read_csv(r"timeseries/dataset_corrections/flower_out.csv")
final= merged_final.merge(flower_correction[['polygon_id','isFlowering','isFruiting']], on='polygon_id', how='left')


final['isFlowering']= final['isFlowering'].fillna('no')
final['isFlowering'].value_counts()
final['isFruiting']= final['isFruiting'].fillna('no')
final['isFruiting'].value_counts()



final['leafing'] = np.where(
    (final['isFlowering'] == 'yes') | 
    (final['isFruiting'] == 'maybe') | 
    (final['isFlowering'] == 'maybe') | 
    (final['isFruiting'] == 'yes'),
    100,
    final['leafing']
)

final['leafing'] = final['leafing'].fillna(final['leafing_predicted'])



final['date'] = pd.to_datetime(final['date'], format='%Y_%m_%d')
final['dayYear'] = final['date'].dt.dayofyear
final['year'] = final['date'].dt.year
final['date_num'] = (final['date'] - final['date'].min()).dt.days


final.to_csv(r"timeseries/dataset_extracted/dipteryx.csv")


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import pickle
import ruptures as rpt
import statsmodels.api as sm
from scipy.signal import savgol_filter
import seaborn as sns
import geopandas as gpd
import os
import geopandas as gpd
from timeseries.utils import generate_leafing_pdf, customFlowering, customLeafing, customFloweringNumeric
import rasterio
from shapely import box
import shapely
from rasterio.mask import mask
from statistics import mode
from PIL import Image

data= pd.read_csv(r"timeseries/dataset_predictions/dipteryx_sgbt.csv")
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
orthomosaic_list=os.listdir(orthomosaic_path)
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")

with open(r'timeseries/models/xgb_model.pkl', 'rb') as file:
      model = pickle.load(file)
with open(r'timeseries/models/xgb_model_flower.pkl', 'rb') as file:
      model_flower = pickle.load(file)

X=data[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]

Y= data[['area', 'score', 'tag', 'GlobalID', 'iou',
       'date', 'latin', 'polygon_id']]

X_predicted=model.predict(X)
X_predict_flower= model_flower.predict(X)

df_final = Y.copy()  # Copy Y to keep the same structure
df_final['leafing_predicted'] = X_predicted
df_final['isFlowering_predicted'] = X_predict_flower

#lets stem and create flower dataset instances for labeling
flower_dataset= df_final[df_final['isFlowering_predicted']==1]
flower_dataset= flower_dataset.merge(crowns[['polygon_id','geometry']],on='polygon_id',how='left')

path_out= os.path.join(data_path,'flower_dataset')
#extract the features, crown based 
for i, (_, row) in enumerate(flower_dataset.iterrows()):
    print(f"Processing iteration {i + 1} of {len(flower_dataset)}")
    if not os.path.exists(os.path.join(path_out, row['polygon_id']+".png")):
        path_orthomosaic = os.path.join(data_path, 'orthomosaic_aligned_local', f"BCI_50ha_{row['date']}_local.tif")
        try:
            with rasterio.open(path_orthomosaic) as src:
                bounds = row.geometry.bounds
                box_crown_5 = box(bounds[0] - 5, bounds[1] - 5, bounds[2] + 5, bounds[3] + 5)
                print(box_crown_5)
                out_image, out_transform = mask(src, [box_crown_5], crop=True)
                x_min, y_min = out_transform * (0, 0)
                xres, yres = out_transform[0], out_transform[4]

                transformed_geom = shapely.ops.transform(
                        lambda x, y: ((x - x_min) / xres, (y - y_min) / yres),
                        row.geometry
                    )
                
                img_name = f"{row['polygon_id']}.png"
                img_path = os.path.join(path_out, img_name)

                fig, ax = plt.subplots(figsize=(10, 10))

                ax.imshow(out_image.transpose((1, 2, 0))[:, :, 0:3])

                ax.plot(*transformed_geom.exterior.xy, color='red')

                for interior in transformed_geom.interiors:
                    ax.plot(*interior.xy, color='red')

                ax.axis('off')

                fig.savefig(img_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                print(f"Saved: {img_path}")

        except Exception as e:
            print(f"Error processing {row['polygon_id']}: {e}")
    else:
        print("it already exists in dataset")








## transfor the date? for what? 
df_final['date'] = pd.to_datetime(df_final['date'], format='%Y_%m_%d')
df_final['dayYear'] = df_final['date'].dt.dayofyear
df_final['year']= df_final['date'].dt.year
df_final['date_num']= (df_final['date'] -df_final['date'].min()).dt.days


#lets bring in the actual labels to clean it up
training_dataset=pd.read_csv(r"timeseries/dataset_training/train_sgbt.csv")
merged_final= df_final.merge(training_dataset[['polygon_id','leafing', 'isFlowerin']], on='polygon_id', how='left')

merged_final['leafing'] = merged_final['leafing'].fillna(merged_final['leafing_predicted'])
merged_final['isFlowering_predicted'] = merged_final['isFlowering_predicted'].replace({0: 'no', 1: 'yes', 2: 'yes'})

merged_final['isFlowerin'] = merged_final['isFlowerin'].fillna(merged_final['isFlowering_predicted'])


data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
orthomosaic_list=os.listdir(orthomosaic_path)
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")
path_out= os.path.join(data_path,'flower_dataset')
#extract the features, crown based 
for i, (_, row) in enumerate(flower_dataset.iterrows()):
    print(f"Processing iteration {i + 1} of {len(flower_dataset)}")
    if not os.path.exists(os.path.join(path_out, row['polygon_id']+".png")):
        path_orthomosaic = os.path.join(data_path, 'orthomosaic_aligned_local', f"BCI_50ha_{row['date']}_local.tif")
        try:
            with rasterio.open(path_orthomosaic) as src:
                bounds = row.geometry.bounds
                box_crown_5 = box(bounds[0] - 5, bounds[1] - 5, bounds[2] + 5, bounds[3] + 5)
                print(box_crown_5)
                out_image, out_transform = mask(src, [box_crown_5], crop=True)
                x_min, y_min = out_transform * (0, 0)
                xres, yres = out_transform[0], out_transform[4]

                transformed_geom = shapely.ops.transform(
                        lambda x, y: ((x - x_min) / xres, (y - y_min) / yres),
                        row.geometry
                    )
                
                img_name = f"{row['polygon_id']}.png"
                img_path = os.path.join(path_out, img_name)

                fig, ax = plt.subplots(figsize=(10, 10))

                ax.imshow(out_image.transpose((1, 2, 0))[:, :, 0:3])

                ax.plot(*transformed_geom.exterior.xy, color='red')

                for interior in transformed_geom.interiors:
                    ax.plot(*interior.xy, color='red')

                ax.axis('off')

                fig.savefig(img_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                print(f"Saved: {img_path}")

        except Exception as e:
            print(f"Error processing {row['polygon_id']}: {e}")
    else:
        print("it already exists in dataset")












plt.figure(figsize=(20, 6))

for tree in merged_final['GlobalID'].unique():
    indv = merged_final[merged_final['GlobalID'] == tree]
    tagn = indv['tag'].unique()[0]
    
    # Line plot for leafing
    plt.plot(indv['date'], indv['leafing_predicted'], label=f"Tree {tagn}", alpha=0.7)
    
    # Scatter plot for flowering
    flowering_points = indv[indv['isFlowerin'] == 'yes']
    plt.scatter(flowering_points['date'], flowering_points['leafing'],
                color='crimson', s=40, marker='*', label=f"Flowering {tagn}", edgecolor='black')

plt.xlabel('Date')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Dipteryx oleifera: Leaf coverage vs Date (Flowering Highlighted)')
plt.legend(title="ForestGeo Tag", bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()

import os
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
ortho=os.path.join(data_path,'orthomosaic_aligned_local')
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")
merged_final.columns
merged_final= merged_final.merge(crowns[['polygon_id','geometry']],on='polygon_id',how='left')
merged_final['date'] = merged_final['date'].dt.strftime('%Y_%m_%d')
from timeseries.utils import generate_leafing_pdf
generate_leafing_pdf(merged_final[merged_final['isFlowering_predicted']=='yes'],r"plots/dipteryx_flowering_predicted.pdf", ortho,crowns_per_page=12, variables=['leafing','isFlowering_predicted'])
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from PIL import Image
data_path=r"C:\Users\Vicente\Documents\Data"

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
labels_path=r"timeseries/export_2025_03_05.csv"
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
orthomosaic_list=os.listdir(orthomosaic_path)

#open gdf containing polygons
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")
labels=pd.read_csv(labels_path)
crowns_labeled= labels.merge(crowns[['area', 'score', 'tag', 'iou', 'geometry','polygon_id']],
                              left_on="polygon_id",
                                right_on="polygon_id",
                                  how="left")

crowns_labeled= crowns_labeled[crowns_labeled['segmentation']=='good']

def custom_leafing(leafing_values):
    values= list(leafing_values)
    sd_values= np.std(values)
    if len(values)==1:
        result = sum(values)/len(values) if values else None
    elif len(values)>= 2 and sd_values <=5:
        result = sum(values)/len(values) if values else None
    elif len(values)>=0 and sd_values>5:
        result = None
    return result

def customFlowering(floweringValues): 
    floweringValues=list(floweringValues)
    if len(floweringValues)==1:
        return floweringValues[0]
    if all(value == floweringValues[0] for value in floweringValues):
            return floweringValues[0]
    if "maybe" in floweringValues:
        return "maybe"
    if "yes" in floweringValues and "no" in floweringValues:
        return None


crowns_labeled_avg = crowns_labeled.groupby("polygon_id").agg({
    "leafing": custom_leafing,
    "latin": "first",
    "isFlowering": customFlowering, 
    "geometry": "first",
    "date":"first",
    "area":"first",
    "tag":"first",
    "iou":"first",
    "score":"first"
}).reset_index()

crowns_labeled_final=crowns_labeled_avg.dropna(subset=['leafing'])

path_out= os.path.join(data_path,"train_dataset")
#extract the features, crown based 
for i, (_, row) in enumerate(crowns_labeled_final.iterrows()):
    print(f"Processing iteration {i + 1} of {len(crowns_labeled_final)}")
    path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")
    try:
        with rasterio.open(path_orthomosaic) as src:
            out_image, out_transform = mask(src, [row.geometry], crop=True)
            img_array = np.moveaxis(out_image, 0, -1) 
            img_array = img_array.astype(np.uint8)
            img_name = f"{row['polygon_id']}.png"
            img_path = os.path.join(path_out, img_name)
            Image.fromarray(img_array).save(img_path)
            
            print(f"Saved: {img_path}")

    except Exception as e:
        print(f"Error processing {row['polygon_id']}: {e}")


training_d= crowns_labeled_final[['polygon_id','leafing']]
training_d['filename']= crowns_labeled_final['polygon_id']+".png"
training_d=training_d[['filename','leafing']]
training_d.to_csv('timeseries/train_dataset.csv')
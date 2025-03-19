import os
import pandas as pd
import geopandas as gpd
from timeseries.utils import generate_leafing_pdf, customFlowering, customLeafing, customFloweringNumeric
import rasterio
from shapely import box
import matplotlib.pyplot as plt
import shapely
from rasterio.mask import mask

#load polygons
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")

#load main dataset of labels 
labels_path=r"timeseries/dataset_raw/export_2025_03_11.csv"
labels=pd.read_csv(labels_path)

labels_not_flower= labels[labels['isFlowering']=="no"]

#read the flower dataset
flower_correction=pd.read_csv(r'timeseries/dataset_corrections/flower1.csv')
flower_correction['polygon_id']=flower_correction['globalId']+"_"+flower_correction['date'].str.replace("_","-")

all_data = pd.concat([labels_not_flower, flower_correction], axis=0, ignore_index=True)

#merge both datasets
crowns_labeled= all_data.merge(crowns[['area', 'score', 'tag', 'iou', 'geometry','polygon_id']],
                              left_on="polygon_id",
                                right_on="polygon_id",
                                  how="left")

#data quality
#crowns_labeled= crowns_labeled[(crowns_labeled["segmentation"]=="good")|(crowns_labeled["segmentation"]=="okay")]
crowns_labeled= crowns_labeled[crowns_labeled["segmentation"]=="good"]

                
#saturate with corrections
check0=pd.read_csv(r"timeseries/dataset_corrections/check_0.csv")
check_01=pd.read_csv(r"timeseries/dataset_corrections/check_01.csv")
crowns_final=pd.concat([crowns_labeled,check0,check0,check0,check_01,check_01,check_01]) #saturate the dataset to make the lcustum leafing trigger its conditions

crowns_final['floweringIntensity'] = crowns_final['floweringIntensity'].fillna(0)

crowns_labeled_avg = crowns_final.groupby("polygon_id").agg({
    "leafing": customLeafing,
    "isFlowering":customFlowering,
    "floweringIntensity": customFloweringNumeric,
    "latin": "first",  
    "geometry": "first",
    "date":"first",
    "area":"first",
    "tag":"first",
    "iou":"first",
    "score":"first"
}).reset_index()

crowns_labeled_avg=crowns_labeled_avg[~crowns_labeled_avg['leafing'].isna()]

cnn_dataset= crowns_labeled_avg[['polygon_id','leafing']]
cnn_dataset['polygon_id']=cnn_dataset["polygon_id"]+'.png'
cnn_dataset.to_csv(r'timeseries/dataset_training/train_cnn.csv')

crowns_labeled_avg = gpd.GeoDataFrame(crowns_labeled_avg, geometry='geometry')
crowns_labeled_avg.set_crs("EPSG:32617", allow_override=True, inplace=True)  

#make sure the combined dataset has sound estimates in leafing
print(crowns_labeled_avg['leafing'].describe())

#one comes out for the sgbt dataset
crowns_labeled_avg.to_file(r'timeseries/dataset_training/train.shp')

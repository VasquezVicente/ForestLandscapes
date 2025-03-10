import os
import geopandas as gpd
import pandas as pd
import numpy as np

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


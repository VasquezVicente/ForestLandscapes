import os
import geopandas as gpd
import pandas as pd
import rasterio
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
from timeseries.timeseries_tools import generate_leafing_pdf, customFlowering, customLeafing
import seaborn as sns
from PIL import Image

#PATHS
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
labels_path=r"timeseries/dataset_raw/export_2025_03_11.csv"
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

#i need to my corrections and saturate before aggregating

check0=pd.read_csv(r"timeseries/dataset_corrections/check_0.csv")
check_01=pd.read_csv(r"timeseries/dataset_corrections/check_01.csv")

crowns_final=pd.concat([crowns_labeled,check0,check0,check_01,check_01]) #saturate the dataset to make the lcustum leafing trigger its conditions

crowns_labeled_avg = crowns_final.groupby("polygon_id").agg({
    "leafing": customLeafing,
    "isFlowering":customFlowering,
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


path_out= os.path.join(data_path,"train_dataset")
#extract the features, crown based 
for i, (_, row) in enumerate(crowns_labeled_avg.iterrows()):
    print(f"Processing iteration {i + 1} of {len(crowns_labeled_avg)}")
    if not os.path.exists(os.path.join(path_out, row['polygon_id']+".png")):
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
    else:
        print("it already exists in dataset")


print(crowns_labeled_avg['leafing'].describe())
training_d=crowns_labeled_avg[crowns_labeled_avg['leafing']<=100]

training_d= training_d[['polygon_id','leafing']]
training_d['filename']= training_d['polygon_id']+".png"
training_d=training_d[['filename','leafing']]
training_d.to_csv('timeseries/dataset_training/train_cnn.csv')
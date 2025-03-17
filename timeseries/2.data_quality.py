import os
import pandas as pd
import geopandas as gpd
from timeseries.utils import generate_leafing_pdf, customFlowering, customLeafing
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
crowns_labeled= crowns_labeled[crowns_labeled["segmentation"]=="good"]


#####this is for extracting the flower crowns
crowns_flower = crowns_labeled[(crowns_labeled['isFlowering'] == 'yes') | (crowns_labeled['isFlowering'] == 'maybe')]
path_out= os.path.join(data_path,'flower_dataset')

#extract the features, crown based 
for i, (_, row) in enumerate(crowns_flower.iterrows()):
    print(f"Processing iteration {i + 1} of {len(crowns_flower)}")
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

crowns_flower=crowns_flower.drop(columns=['geometry'])
crowns_flower.to_csv(r"timeseries/dataset_corrections/check_flower1.csv")
######################


                    

#saturate with corrections
check0=pd.read_csv(r"timeseries/dataset_corrections/check_0.csv")
check_01=pd.read_csv(r"timeseries/dataset_corrections/check_01.csv")
crowns_final=pd.concat([crowns_labeled,check0,check0,check0,check_01,check_01,check_01]) #saturate the dataset to make the lcustum leafing trigger its conditions


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

#make sure the combined dataset has sound estimates in leafing
print(crowns_labeled_avg['leafing'].describe())

#one comes out for the sgbt dataset
crowns_labeled_avg.to_file(r'timeseries/dataset_training/train.shp')

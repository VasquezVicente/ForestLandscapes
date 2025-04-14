import os
import pandas as pd
import geopandas as gpd
from timeseries.utils import generate_leafing_pdf, customFlowering, customLeafing, customFloweringNumeric
import rasterio
from shapely import box
import matplotlib.pyplot as plt
import shapely
from rasterio.mask import mask
import numpy as np

#load polygons
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
orthomosaic_list=os.listdir(orthomosaic_path)
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")

#load main dataset of labels 
labels_path=r"timeseries/dataset_raw/export_2025_04_09.csv"
labels=pd.read_csv(labels_path)

#merge both datasets
crowns_labeled= labels.merge(crowns[['area', 'score', 'tag', 'iou', 'geometry','polygon_id']],
                              left_on="polygon_id",
                                right_on="polygon_id",
                                  how="left")

#Keep only good segmentation
crowns_labeled= crowns_labeled[crowns_labeled["segmentation"]=="good"]

#split the dataset to the flowers dataset
flower_dataset=crowns_labeled[(crowns_labeled["isFlowering"]=="yes")|(crowns_labeled["isFlowering"]=="maybe")]
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


flower_csv= flower_dataset.drop(columns=['geometry'])
flower_csv= flower_csv[['isFlowering', 'leafing', 'floweringIntensity',
       'segmentation', 'observation_id', 'polygon_id', 'date', 'globalId',
       'latin', 'area', 'score', 'tag', 'iou']]
flower_csv.to_csv(r'timeseries/dataset_corrections/flower.csv')



non_flower= crowns_labeled[crowns_labeled['isFlowering']=="no"]
#read the flower dataset back in 
flower_dataset=pd.read_csv(r"timeseries/dataset_corrections/flower_out.csv")
# get rid of the flowering lianas
flower_dataset= flower_dataset[flower_dataset['flowering_liana']!="yes"]
flower_dataset= flower_dataset.merge(crowns[['latin','polygon_id','geometry','date']],on='polygon_id', how='left')


#generate a pdf for dipteryx oleifera
dipteryx_flowering= flower_dataset[flower_dataset['latin']=="Dipteryx oleifera"]
generate_leafing_pdf(dipteryx_flowering, r'plots/dipteryx_flowering.pdf',orthomosaic_path, crowns_per_page=12,variables=['floweringIntensity','isFlowering'] )
jacaranda_flowering= flower_dataset[flower_dataset['latin']=="Jacaranda copaia"]
generate_leafing_pdf(jacaranda_flowering, r'plots/jacaranda_flowering.pdf',orthomosaic_path, crowns_per_page=12,variables=['floweringIntensity','isFlowering'] )
zanth_flowering= flower_dataset[flower_dataset['latin']=="Zanthoxylum ekmanii"]
generate_leafing_pdf(zanth_flowering, r'plots/zanthoxylum_flowering.pdf',orthomosaic_path, crowns_per_page=12,variables=['floweringIntensity','isFlowering'] )
globu_flowering= flower_dataset[flower_dataset['latin']=="Symphonia globulifera"]
generate_leafing_pdf(globu_flowering, r'plots/globu_flowering.pdf',orthomosaic_path, crowns_per_page=12,variables=['floweringIntensity','isFlowering'] )
cava_flowering= flower_dataset[flower_dataset['latin']=="Cavanillesia platanifolia"]
generate_leafing_pdf(cava_flowering, r'plots/cava_flowering.pdf',orthomosaic_path, crowns_per_page=12,variables=['floweringIntensity','isFlowering','isFruiting','date'] )


jacaranda_flowering.loc[
    jacaranda_flowering['isFlowering'].isin(['yes', 'maybe']), 'leafing'
] = 100

# For dipteryx
dipteryx_flowering.loc[
    dipteryx_flowering['isFlowering'].isin(['yes', 'maybe']), 'leafing'
] = 100

flower_dataset= pd.concat([dipteryx_flowering, jacaranda_flowering])

#add correction
correction1=pd.read_csv(r'timeseries/dataset_corrections/check_01.csv')
correction1['isFlowering']= "no"
correction0=pd.read_csv(r'timeseries/dataset_corrections/check_0.csv')
correction0['isFlowering']= "no"

crowns_final= pd.concat([non_flower,correction0, correction1,correction0, correction1, flower_dataset])

from statistics import mode
def customLeafing(leafing_values):
    values = list(leafing_values)
    sd_values = np.std(values)
    if len(values) == 1:
        return values[0]
    if len(values) >= 2 and sd_values <= 5:
        result= sum(values) / len(values)
        return result
    if len(values) >= 2 and sd_values > 5:
        try:
            reference_value = mode(values)
        except:
            reference_value = np.median(values)
        filtered_values = [v for v in values if abs(v - reference_value) <= 5]
        if filtered_values:
            result=sum(filtered_values) / len(filtered_values)
            return result
        else:
            result= None
            return result

def customFloweringNumeric(floweringN):
    values = list(floweringN)
    sd_values = np.std(values)
    if len(values) == 1:
        return values[0]
    if len(values) >= 2 and sd_values <= 5:
        result= sum(values) / len(values)
        return result
    if len(values) >= 2 and sd_values > 5:
        try:
            reference_value = mode(values)
        except:
            reference_value = np.median(values)
        filtered_values = [v for v in values if abs(v - reference_value) <= 5]
        if filtered_values:
            result=sum(filtered_values) / len(filtered_values)
            return result
        else:
            result= None
            return result

def customFlowering(floweringValues): 
    floweringValues=list(floweringValues)
    if len(floweringValues)==1:
        print("Only one isFlowering value")
        return floweringValues[0]
    if all(value == floweringValues[0] for value in floweringValues):
        print("All values are the same: ", floweringValues[0])   
        return floweringValues[0]         
    if "maybe" in floweringValues:
        print("maybe is present")
        return "maybe"
    if "yes" in floweringValues and "no" in floweringValues:
        print("contradicting yes/no")
        return "maybe"
    else:
        print('no case')

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


#bring in the species once more

crowns_labeled_avg= crowns_labeled_avg.merge(crowns[['latin','polygon_id','geometry']], left_on='polygon_id',right_on='polygon_id', how='left')
crowns_labeled_avg['latin'] = crowns_labeled_avg['latin_x'].combine_first(crowns_labeled_avg['latin_y'])
crowns_labeled_avg['geometry'] = crowns_labeled_avg['geometry_x'].combine_first(crowns_labeled_avg['geometry_y'])
crowns_labeled_avg.drop(columns=['latin_x', 'latin_y','geometry_x','geometry_y'], inplace=True)


#i only want to keep dipteryx and jacaranda for my cnn of flowers
flower_cnn= crowns_labeled_avg[(crowns_labeled_avg['latin']=='Jacaranda copaia')|(crowns_labeled_avg['latin']=='Dipteryx oleifera')]

def classify(row):
    # If not flowering
    if row['isFlowering'] == 'no':
        return 0

    # If flowering (yes or maybe) and species is Dipteryx
    elif row['isFlowering'] in ['yes', 'maybe'] and row['latin'] == 'Dipteryx oleifera':
        return 1

    # If flowering (yes or maybe) and species is Jacaranda
    elif row['isFlowering'] in ['yes', 'maybe'] and row['latin'] == 'Jacaranda copaia':
        return 2

    else:
        print(row['isFlowering'], row['latin'])
        return None  

flower_cnn['class'] = flower_cnn.apply(classify, axis=1)
flower_cnn['date'] = flower_cnn['polygon_id'].str.split('_').str[1]
flower_cnn['date'] = flower_cnn['date'].str.replace('-', '_')

flower_cnn[flower_cnn['class'].isna()]

orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
#list of orthomosaics
orthomosaic_list=os.listdir(orthomosaic_path)
import numpy as np
from PIL import Image
path_out=os.path.join(data_path,"flower_data")
os.makedirs(path_out, exist_ok=True)
for i, (_, row) in enumerate(flower_cnn.iterrows()):
    print(f"Processing iteration {i + 1} of {len(flower_cnn)}")
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

flower_cnn['file']= flower_cnn['polygon_id']+".png"
flower_cnn[['file','class']].to_csv(r'timeseries/dataset_training/train_cnn_flower.csv')



crowns_labeled_avg=crowns_labeled_avg[~crowns_labeled_avg['leafing'].isna()]


cnn_dataset= crowns_labeled_avg[['polygon_id','leafing']]
cnn_dataset['polygon_id']=cnn_dataset["polygon_id"]+'.png'

cnn_dataset.to_csv(r'timeseries/dataset_training/train_cnn.csv')

sgbt_dataset= crowns_labeled_avg[['polygon_id','leafing','isFlowering']].merge(crowns[['date','geometry','polygon_id']],
                              left_on="polygon_id",
                                right_on="polygon_id",
                                  how="left")

sgbt_dataset = gpd.GeoDataFrame(sgbt_dataset, geometry='geometry')
sgbt_dataset.set_crs("EPSG:32617", allow_override=True, inplace=True)  

#make sure the combined dataset has sound estimates in leafing
print(sgbt_dataset['leafing'].describe())

#one comes out for the sgbt dataset
sgbt_dataset.to_file(r'timeseries/dataset_training/train.shp')

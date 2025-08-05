import pandas as pd
import os
import geopandas as gpd
import pickle

data= pd.read_csv(r"timeseries/dataset_predictions/quararibea_sgbt.csv")    ##contains the sgbt features
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"    #Not needed
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")            #Not needed
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp") #Not needed
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")   #Not needed
orthomosaic_list=os.listdir(orthomosaic_path) #Not needed
crowns=gpd.read_file(path_crowns) #Not needed
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")  #Not needed

with open(r'timeseries/models/xgb_model.pkl', 'rb') as file:
      model = pickle.load(file)
#with open(r'timeseries/models/xgb_model_flower.pkl', 'rb') as file:
#      model_flower = pickle.load(file)

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
merged_final= df_final.merge(training_dataset[['polygon_id','leafing', 'isFlowerin']], on='polygon_id', how='left')

merged_final['isFlowerin'].isna().sum()
merged_final['isFlowering_predicted'] = merged_final['isFlowering_predicted'].replace({0: 'no', 1: 'yes', 2: 'yes'})
merged_final['isFlowerin']= merged_final['isFlowerin'].fillna(merged_final['isFlowering_predicted'])

merged_final['isFlowerin'] = np.where(
    merged_final['isFlowerin'] == 'no',
    'no',
    np.where(
        merged_final['isFlowerin'] == 'yes',
        'yes',
        np.where(
            pd.to_numeric(merged_final['isFlowerin'], errors='coerce') > 0,
            'yes',
            'no'
        )
    )
)

merged_final['leafing'] = np.where(
    merged_final['isFlowerin'] == 'yes',
    100,
    merged_final['leafing']
)

merged_final['leafing'] = merged_final['leafing'].fillna(merged_final['leafing_predicted'])



merged_final['date'] = pd.to_datetime(merged_final['date'], format='%Y_%m_%d')
merged_final['dayYear'] = merged_final['date'].dt.dayofyear
merged_final['year']= merged_final['date'].dt.year
merged_final['date_num']= (merged_final['date'] -merged_final['date'].min()).dt.days



merged_final.to_csv(r"timeseries/dataset_extracted/quararibea.csv")


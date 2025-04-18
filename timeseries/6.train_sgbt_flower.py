import pandas as pd
from sklearn.datasets import make_classification
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patches as patches
from shapely.affinity import affine_transform
import pickle
from xgboost import XGBClassifier

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")

training_dataset=gpd.read_file(r"timeseries/dataset_training/train_sgbt.csv")


columns_to_convert = ['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM','shadowM','rSD', 'gSD', 'bSD',
                      'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD','leafing']

for col in columns_to_convert:
    training_dataset[col] = pd.to_numeric(training_dataset[col], errors='coerce')

# Now check the datatypes
print(training_dataset.dtypes)
print(training_dataset.columns)

#lets do only the flowering dipteryx and jacaranda dipteryx
flower_dataset= training_dataset.merge(crowns[['polygon_id','latin']], on="polygon_id", how='left')

flower_dataset=flower_dataset[(flower_dataset['latin']=="Dipteryx oleifera")|(flower_dataset['latin']=='Jacaranda copaia')]

flower_dataset['isFlowerin'] = flower_dataset['isFlowerin'].replace('maybe', 'yes')

X=flower_dataset[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]
       
def custom_label(row):
    if row['isFlowerin'] == 'no':
        return 0
    elif row['isFlowerin'] == 'yes' and row['latin'] == 'Dipteryx oleifera':
        return 1
    elif row['isFlowerin'] == 'yes' and row['latin'] == 'Jacaranda copaia':
        return 2
    else:
        return 0  # fallback to 0 for any unmatched "yes" + other species

# Apply to dataset
y = flower_dataset.apply(custom_label, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = XGBClassifier(
    n_estimators=5000,
    max_depth=20,
    learning_rate=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

with open(r'timeseries/models/xgb_model_flower.pkl', 'wb') as file:
    pickle.dump(model, file)
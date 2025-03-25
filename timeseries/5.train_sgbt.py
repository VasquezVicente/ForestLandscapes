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

training_dataset=gpd.read_file(r"timeseries/dataset_training/train_sgbt.csv")


columns_to_convert = ['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM','shadowM','rSD', 'gSD', 'bSD',
                      'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD','leafing']

for col in columns_to_convert:
    training_dataset[col] = pd.to_numeric(training_dataset[col], errors='coerce')

# Now check the datatypes
print(training_dataset.dtypes)

training_dataset=training_dataset[~training_dataset['leafing'].isna()]
#split the features and output
X=training_dataset[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]
y=training_dataset[['leafing']]

#split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = XGBRegressor(
    n_estimators=5000,
    max_depth=20, 
    learning_rate=0.001, 
    subsample=0.8,  
    colsample_bytree=0.8, 
    random_state=42,
    n_jobs=-1 
)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# Predict on test set
y_pred = model.predict(X_test)


# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal line')  # Line for perfect prediction
plt.xlabel('True Values (Leafing)')
plt.ylabel('Predicted Values (Leafing)')
plt.title('True vs Predicted Leafing Values')
plt.legend()
plt.grid(True)
plt.show()

with open(r'timeseries/models/xgb_model.pkl', 'wb') as file:
    pickle.dump(model, file)



training_dataset.loc[y_test.index, 'leafing_predicted'] = y_pred
training_dataset=training_dataset[~training_dataset['leafing_predicted'].isna()]
training_dataset['bias']= training_dataset['leafing_predicted']-training_dataset['leafing']
dataset_bias= training_dataset[training_dataset['bias']>abs(5)]
dataset_bias=dataset_bias.sort_values('bias')
print(dataset_bias)
dataset_bias.to_csv(r'timeseries/dataset_results/bias_sgbt.csv')

from timeseries.utils import generate_leafing_pdf
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
ortho=os.path.join(data_path,'orthomosaic_aligned_local')

generate_leafing_pdf(dataset_bias,r'plots/check1.pdf',orthomosaic_path=ortho, crowns_per_page= 12, variables=['leafing_predicted','leafing'])
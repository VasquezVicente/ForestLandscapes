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



training_dataset=gpd.read_file(r"timeseries/training_dataset.shp")
print(training_dataset.columns)



#split the features and output
X=training_dataset[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD']]
y=training_dataset[['leafing']]

#split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


model = XGBRegressor(
    n_estimators=4000,
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


plt.plot(y_test, y_pred, 'o')  # 'o' will plot points as circles
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal line')  # Line for perfect prediction
plt.xlabel('True Values (Leafing)')
plt.ylabel('Predicted Values (Leafing)')
plt.title('True vs Predicted Leafing Values')
plt.legend()
plt.grid(True)
plt.show()

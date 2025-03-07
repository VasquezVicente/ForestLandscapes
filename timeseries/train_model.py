import pandas as pd
from sklearn.datasets import make_classification
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import rasterio
from rasterio.mask import mask
import os
import shapely
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patches as patches
from shapely.affinity import affine_transform
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

training_dataset=gpd.read_file(r"timeseries/training_dataset.shp")
print(training_dataset.columns)

#split the features and output
X=training_dataset[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD']]
y=training_dataset[['leafing']]

#split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


model = XGBRegressor(
    objective="reg:squarederror", 
    n_estimators=4000,
    max_depth=20, 
    learning_rate=0.005, 
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

import matplotlib.pyplot as plt
plt.plot(y_test, y_pred, 'o')  # 'o' will plot points as circles
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()

training_dataset.loc[X_test.index, 'predicted_leafing'] = y_pred

visualize_test = training_dataset[~pd.isna(training_dataset['predicted_leafing'])]

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
orthomosaic_list=os.listdir(orthomosaic_path)
crowns_per_page = 12
crowns_plotted = 0

with PdfPages("plots/visualize_results.pdf") as pdf_pages:
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()

    for i, (_, row) in enumerate(visualize_test.iterrows()):
        path_orthomosaic = os.path.join(orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")

        with rasterio.open(path_orthomosaic) as src:
            bounds = row.geometry.bounds
            box_crown_5 = box(bounds[0]-5, bounds[1]-5, bounds[2]+5, bounds[3]+5)

            out_image, out_transform = mask(src, [box_crown_5], crop=True)
            x_min, y_min = out_transform * (0, 0)
            xres, yres = out_transform[0], out_transform[4]

            # Transform geometry
            transformed_geom = shapely.ops.transform(
                lambda x, y: ((x - x_min) / xres, (y - y_min) / yres),
                row.geometry
            )
 
            ax = axes[crowns_plotted % crowns_per_page]
            ax.imshow(out_image.transpose((1, 2, 0))[:, :, 0:3])
            ax.plot(*transformed_geom.exterior.xy, color='red', linewidth=2)
            ax.axis('off')

            # Add text label
            latin_name = row['latin']
            numeric_feature_1 = row['leafing']
            numeric_feature_2 = row['predicted_leafing']
            ax.text(5, 5, f"{latin_name}\nLeafing: {numeric_feature_1}\pLeaf: {numeric_feature_2}",
                    fontsize=12, color='white', backgroundcolor='black', verticalalignment='top')

            crowns_plotted += 1

        # Save PDF and start a new page every 12 crowns
        if crowns_plotted % crowns_per_page == 0 or i == len(visualize_test) - 1:
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)

            # Create new figure for the next batch
            if i != len(visualize_test) - 1:  # Prevent unnecessary re-creation at end
                fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                axes = axes.flatten()


            elif transformed_geom.geom_type == "MultiPolygon":
                continue

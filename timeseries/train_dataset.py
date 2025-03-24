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
import seaborn as sns
from PIL import Image

#PATHS
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
#list of orthomosaics
orthomosaic_list=os.listdir(orthomosaic_path)

crowns_labeled_avg= gpd.read_file(r'timeseries/dataset_training/train.shp')
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

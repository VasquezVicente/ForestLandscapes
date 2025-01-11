
from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
import rasterio


site_path = "/content/drive/Shareddrives/detectree2/data/BCI_50ha"
img_path = site_path + "/rgb/2015.06.10_07cm_ORTHO.tif"
tiles_path = site_path + "/tilespred/"

# Location of trained model
model_path = "/content/drive/Shareddrives/detectree2/models/220629_ParacouSepilokDanum_JB.pth"

# Specify tiling
buffer = 30
tile_width = 40
tile_height = 40
tile_data(img_path, tiles_path, buffer, tile_width, tile_height, dtype_bool = True)
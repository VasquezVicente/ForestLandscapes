#detect py serves the purpose of detecting the crown of the tree over the image
#installing detectree2 and its complexities is difficult task as it often fails to install
#git clone a temporary file and install it from there 
#git clone https://github.com/PatBall1/detectree2.git
import os 
import json
from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
import rasterio

config_path = "crown-segment/config.json"
def detect_crowns(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    tiles_path = os.path.join(config["site_path"], "tilespred/")
    os.makedirs(tiles_path, exist_ok=True)
    #setup the model configuration
    cfg = setup_cfg(update_model=config["model_path"])
    #tile the image
    tile_data(config["image_path"], tiles_path, config["buffer"], config["tile_width"], config["tile_height"], dtype_bool = True)
    predict_on_data(tiles_path, predictor=DefaultPredictor(cfg))
    project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")
    crowns= stitch_crowns(tiles_path + "predictions_geo", 1.0)
    clean = clean_crowns(crowns, config["iou_th"], confidence=0)
    clean = clean[clean["Confidence_score"] > config["confidence_score"]]
    clean = clean.set_geometry(clean.simplify(config["simplify_tolerance"]))
    clean.crs = "EPSG:32617"
    clean.to_file(os.path.join(config["site_path"], config["image_path"]+"predicted.shp"))
    print("Crown detection complete")






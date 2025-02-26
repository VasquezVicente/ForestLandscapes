import os
import pandas as pd
import json

location=r"timeseries/export_2025_02_25"
files= os.listdir(location)

with open(os.path.join(location,files[0]), "r") as f:
    observations = json.load(f)  

with open(os.path.join(location, files[1]), "r") as f:
    plants=json.load(f)

data = []
for obs in observations:
    path_items = obs.get("__key__", {}).get("path", "").replace('"', '').split(", ")
    second_item = path_items[1] if len(path_items) > 1 else None  
    data.append({
        "isFlowering": obs.get("isFlowering"),
        "leafing": obs.get("leafing"),
        "floweringIntensity": obs.get("floweringIntensity"),
        "segmentation": obs.get("segmentation"),
        "observation_id": obs.get("__key__", {}).get("name"),
        "polygon_id": second_item,
    })

df = pd.DataFrame(data)


print(json.dumps(plants, indent=4)) 
data_plants= []
for obs in plants:
    data_plants.append({
        "date": obs.get("date"),
        "globalId": obs.get("globalId"),
        "latin": obs.get("latinName"),
        "polygon_id":  obs.get("__key__", {}).get("name"),
    })

df_plants= pd.DataFrame(data_plants)

df_merged= df.merge(df_plants, left_on='polygon_id', right_on='polygon_id', how='left')

#export as csv
df_merged.to_csv("timeseries/50ha_timeseries_labels.csv")
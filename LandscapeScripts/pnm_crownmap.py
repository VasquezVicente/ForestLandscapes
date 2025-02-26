import os
import geopandas as gpd
import rasterio
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from pillow_heif import register_heif_opener

path_images=r"C:\Users\Vicente\Downloads\Sanlorenzo-20250121T141322Z-001\Sanlorenzo"

list_images=os.listdir(path_images)

register_heif_opener()

image= Image.open(os.path.join(path_images,list_images[0]))

image.tags

def get_date_taken(image_path):
    image = Image.open(image_path)
    exif_data = image.getexif()
    if exif_data:
        exif_dict = {TAGS.get(tag): value for tag, value in exif_data.items()}
        return exif_dict.get('DateTime')
    return None

data = []

for image_name in list_images:
    image_path = os.path.join(path_images, image_name)
    date_taken = get_date_taken(image_path)
    data.append({'image_name': image_name, 'date_time': date_taken})

df = pd.DataFrame(data)
df.to_csv(r"timeseries\images_dates.csv")
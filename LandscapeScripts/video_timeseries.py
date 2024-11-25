#python
#generate 4k video timeseries
import os
import glob
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box
import cv2
import numpy as np

# list of images tif
wd= r'\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone'
orthomosaics = glob.glob(os.path.join(wd, '**', '*orthomosaic.tif'), recursive=True)
ortho_df = pd.DataFrame(orthomosaics, columns=['orthomosaic'])
ortho_df['mission'] = ortho_df['orthomosaic'].apply(lambda x: str(x.split('\\')[7]))
def extract_date(mission):
    try:
        date_str = '_'.join(mission.split('_')[2:5])
        return pd.to_datetime(date_str, format='%Y_%m_%d')
    except ValueError:
        return None
ortho_df['date'] = ortho_df['mission'].apply(extract_date)
ortho_df = ortho_df.dropna(subset=['date'])
print(ortho_df)

# I need only the images from 2022 dec and onwards
ortho_df = ortho_df[ortho_df['date'] >= pd.to_datetime('2022-12-01')]
# I need only the 50 ha plot images
ortho_df = ortho_df[ortho_df['mission'].str.contains('50ha')]

# pull the 50ha shapefile
BCI_50ha_shapefile = r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\aux_files\BCI_Plot_50ha.shp"
BCI_50ha_shapefile= gpd.read_file(BCI_50ha_shapefile)
BCI_50ha_shapefile= BCI_50ha_shapefile.to_crs('EPSG:32617')
shape=BCI_50ha_shapefile.iloc[0]
shape= box(shape.geometry.bounds[0]+150, shape.geometry.bounds[1]+100, shape.geometry.bounds[2]-150, shape.geometry.bounds[3]-100)


import os
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import numpy as np

# Define paths and parameters
width, height = 3840, 2160
outpath = r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\cropped"
outpath2 = r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\video2"


if not os.path.exists(outpath):
    os.makedirs(outpath)

if not os.path.exists(outpath2):
    os.makedirs(outpath2)

# Iterate over each orthomosaic
for ortho_path in ortho_df['orthomosaic']:
    print(ortho_path)
    filename = os.path.basename(ortho_path)
    out_path = os.path.join(outpath, filename)
    out_path2 = os.path.join(outpath2, filename)

    if os.path.exists(out_path2):
        print('File exists')
        continue
    else:
        print('File does not exist')
        with rasterio.open(ortho_path) as src:
            # Crop the image
            out_image, out_transform = mask(src, [shape], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "count": 4
            })

            # Convert to uint8
            if out_meta['dtype'] == 'uint16':
                out_image_normalized = (out_image / 65535.0) * 255.0
                out_image_uint8 = out_image_normalized.astype('uint8')
            else:
                out_image_uint8 = out_image.astype('uint8')

            # Only the first 4 bands
            out_image_uint8 = out_image_uint8[:4]

            # Calculate new dimensions to maintain aspect ratio
            original_height, original_width = out_image_uint8.shape[1:3]
            aspect_ratio = original_width / original_height
            new_width = width
            new_height = int(new_width / aspect_ratio)

            # Resample the image
            data_resampled = np.zeros((out_image_uint8.shape[0], new_height, new_width), dtype='uint8')
            for band in range(out_image_uint8.shape[0]):
                data_resampled[band] = rasterio.warp.reproject(
                    source=out_image_uint8[band],
                    destination=np.empty((new_height, new_width), dtype='uint8'),
                    src_transform=out_transform,
                    src_crs=src.crs,
                    dst_transform=out_transform * out_transform.scale(
                        (original_width / new_width),
                        (original_height / new_height)
                    ),
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )[0]

            out_meta.update({
                "height": new_height,
                "width": new_width,
                "transform": out_transform * out_transform.scale(
                    (original_width / new_width),
                    (original_height / new_height)
                ),
                "dtype": 'uint8'
            })

            # Write the final image
            with rasterio.open(out_path2, "w", **out_meta) as dest:
                dest.write(data_resampled)

            print("Processing and resampling completed for", filename)
# Code to combine the images into a video

output_folder = r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
video_name = 'output_video.mp4'
video_path = os.path.join(output_folder, video_name)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, 2, (new_width, new_height))
# Add images to the video
for filename in sorted(os.listdir(outpath2)):
    image = cv2.imread(os.path.join(outpath2, filename))
    date = filename.split('_')[2:5]
    date = '-'.join(date)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    color = (255, 255, 255) 
    thickness = 5
    position = (50, 100)  
    cv2.putText(image, date, position, font, font_scale, color, thickness, cv2.LINE_AA)
    video.write(image)

# Release the video writer
video.release()

with rasterio.open(r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\video2\BCI_50ha_2024_07_22_orthomosaic.tif")as src:
    print(src.meta)




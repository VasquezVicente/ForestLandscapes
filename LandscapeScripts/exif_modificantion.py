import os 
import pandas as pd
import exifread
import piexif
from PIL import Image
from GPSPhoto import gpsphoto
import tifftools
import pyexiv2
import os
import exifread

path=r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\BCI_50ha_2024_03_06_M3E\DJI_202403061013_001_BCI-AVA\tagged_VUELO1_RGB"


jpegs= sorted([f for f in os.listdir(path) if f.endswith('D.JPG')])
ms_g= sorted([f for f in os.listdir(path) if f.endswith('MS_G.TIF')])
ms_nir= sorted([f for f in os.listdir(path) if f.endswith('MS_NIR.TIF')])
ms_red= sorted([f for f in os.listdir(path) if f.endswith('MS_RE.TIF')])
ms_r= sorted([f for f in os.listdir(path) if f.endswith('MS_R.TIF')])

len(jpegs), len(ms_g), len(ms_nir), len(ms_red), len(ms_r)

# Example usage

for i in range(len(jpegs)):
    i=1
    jpeg_path = os.path.join(path, jpegs[i])

    data= pyexiv2.Image(jpeg_path)
    metadata = data.read_exif()
    latitude = metadata['Exif.GPSInfo.GPSLatitude']
    longitude = metadata['Exif.GPSInfo.GPSLongitude']
    altitude = metadata['Exif.GPSInfo.GPSAltitude']
    print(latitude, longitude, altitude)

    tiff_path_ms_g = os.path.join(path, ms_g[i])
    data_ms_g= pyexiv2.Image(tiff_path_ms_g)
    metadata_ms_g = data_ms_g.read_exif()

    dict= {'Exif.GPSInfo.GPSLatitude': latitude,
              'Exif.GPSInfo.GPSLongitude': longitude, 'Exif.GPSInfo.GPSAltitude': altitude}
    data_ms_g.modify_exif(dict)
    data_ms_g.close()
    data_ms_g2= pyexiv2.Image(tiff_path_ms_g)
    dict2= data_ms_g2.read_exif()
    latitude2 = dict2['Exif.GPSInfo.GPSLatitude']
    longitude2 = dict2['Exif.GPSInfo.GPSLongitude']
    altitude2 = dict2['Exif.GPSInfo.GPSAltitude']
    print(latitude2, longitude2, altitude2)
    


    tiff_path_ms_nir = os.path.join(path, ms_nir[i])
    data_ms_nir= pyexiv2.Image(tiff_path_ms_nir)
    metadata_ms_nir = data_ms_nir.read_exif()
    data_ms_nir.modify_exif(dict)
    dict3= data_ms_nir.read_exif()
    data_ms_nir.close()

    tiff_path_ms_red = os.path.join(path, ms_red[i])
    data_ms_red= pyexiv2.Image(tiff_path_ms_red)
    metadata_ms_red = data_ms_red.read_exif()
    data_ms_red.modify_exif(dict)
    dict4= data_ms_red.read_exif()
    data_ms_red.close()

    tiff_path_ms_r = os.path.join(path, ms_r[i])
    data_ms_r= pyexiv2.Image(tiff_path_ms_r)
    metadata_ms_r = data_ms_r.read_exif()
    data_ms_r.modify_exif(dict)
    dict5= data_ms_r.read_exif()
    data_ms_r.close()

    print("finishing", i+1, "of", len(jpegs))






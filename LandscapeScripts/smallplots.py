import os 
import pandas as pd
import shutil
import Metashape
from collections import defaultdict

project_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\Yasuni_Phantom_20210811\ECUADOR_yasuni_2021_08_11_P4P.psx"

doc = Metashape.Document()
doc.open(project_path)

chunk=doc.chunk

out_path= os.path.join(os.path.dirname(project_path), "RGB")
os.makedirs(out_path, exist_ok=True)

used_names = defaultdict(int)
for photo in chunk.cameras:
    src = photo.photo.path
    base = os.path.basename(src)
    
    count = used_names[base]
    if count == 0:
        dst_name = base
    else:
        name, ext = os.path.splitext(base)
        dst_name = f"{name}_{count}{ext}"
    
    used_names[base] += 1
    dst = os.path.join(out_path, dst_name)
    shutil.copy(src, dst)

from skimage import io, exposure
ref_image = r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\Yasuni_Phantom_20210811\RGB\DJI_0137_1.JPG"
ref_image= io.imread(ref_image)
#io.imread(os.path.join(out_path,"DJI_0053.JPG"))

# Folder of dark images

output_folder = out_path.replace('RGB',"normalized")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(out_path):
    if filename.lower().endswith(('.jpg')):
        img_path = os.path.join(out_path, filename)
        img = io.imread(img_path)

        matched = exposure.match_histograms(img, ref_image, channel_axis=-1)

        out_file = os.path.join(output_folder, filename)
        io.imsave(out_file, matched.astype('uint8'), quality=100)




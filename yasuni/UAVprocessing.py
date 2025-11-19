import os 
import geopandas as gpd
import pandas as pd
import shutil
import Metashape
from collections import defaultdict

project_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV"
dirs=[d for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path,d)) and d.startswith("ECUADOR_yasuni_")]

for idx, d in enumerate(dirs, 1):
    print(f"{idx}: {d}")

choice = int(input("Enter the number of the directory to process: ")) - 1
selected_dir = dirs[choice]
project_path = r"D:\Yasuni\ECUADOR_yasuni_2019_10_06_P4P\ECUADOR_yasuni_2019_10_06_P4P.psx"

doc = Metashape.Document()
doc.open(project_path)
chunk = doc.chunks[0]

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


list_projects= [r"D:\Yasuni\ECUADOR_yasuni_2019_10_06_P4P\ECUADOR_yasuni_2019_10_06_P4P.psx",
                r"D:\Yasuni\ECUADOR_yasuni_2019_09_30_P4P\ECUADOR_yasuni_2019_09_30_P4P.psx",
                r"D:\Yasuni\ECUADOR_yasuni_2019_07_23_P4P\ECUADOR_yasuni_2019_07_23_P4P.psx"
                ]

for project_path in list_projects:
    doc = Metashape.Document()
    doc.open(project_path)
    chunk = doc.chunks[0]
    chunk.matchPhotos(downscale=0,keypoint_limit = 40000, tiepoint_limit = 4000, generic_preselection = True, reference_preselection = True)
    doc.save()
    chunk.alignCameras(adaptive_fitting=True)
    doc.save()
    print(f"Processed project: {project_path}") #break for GCPs

for project_path in list_projects:
    doc = Metashape.Document()
    doc.open(project_path)
    chunk = doc.chunks[0]
    chunk.buildDepthMaps(downscale = 4, filter_mode = Metashape.ModerateFiltering)
    doc.save()

    has_transform = chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation

    if has_transform:
                chunk.buildPointCloud()
                doc.save()

                chunk.buildDem(source_data=Metashape.PointCloudData)
                doc.save()

                chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
                doc.save()

#create an orthoprojection for export
    doc.save()
    print('Processing finished')


from skimage import io, exposure
ref_image = r"D:\Yasuni\ECUADOR_yasuni_2019_07_02_P4P\RGB\2019_Jul_02_Phantom_Flight_4_290.JPG"
ref_image= io.imread(ref_image)
#io.imread(os.path.join(out_path,"DJI_0053.JPG"))

# Folder of dark images
output_folder = out_path.replace('RGB', "normalized")
os.makedirs(output_folder, exist_ok=True)

jpg_files = [f for f in os.listdir(out_path) if f.lower().endswith('.jpg')]
total_files = len(jpg_files)

for i, filename in enumerate(jpg_files, 1):
    img_path = os.path.join(out_path, filename)
    img = io.imread(img_path)

    matched = exposure.match_histograms(img, ref_image, channel_axis=-1)

    out_file = os.path.join(output_folder, filename)
    io.imsave(out_file, matched.astype('uint8'), quality=100)

    print(f"[{i}/{total_files}] Processed: {filename}")


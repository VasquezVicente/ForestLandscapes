import os   #archivos del sistema
import pandas as pd   #data frames
import shutil   #copiar y mover archivos

#mavic flight main folders
images_dir=r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\BCI_50ha_2024_05_07_M3E"
path=os.path.dirname(images_dir)
mission=os.path.basename(images_dir)
folders = [folder for folder in os.listdir(images_dir) if folder.startswith('DJI') and os.path.isdir(os.path.join(images_dir, folder))]

#create a multispectral, rgb and a georefereced folder for each of the three first folders
for folder in folders[0:len(folders)]:
    print(f'Processing {folder}')
    multispectral_folder = os.path.join(path,mission, folder, 'Multispectral')
    rgb_folder = os.path.join(path,mission,folder, 'RGB')
    georeference_folder = os.path.join(path,mission, folder, 'Georeference')

    os.makedirs(multispectral_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(georeference_folder, exist_ok=True)

    
    all_files = os.listdir(os.path.join(path,mission, folder))
    tif_files = [os.path.join(path,mission, folder, f) for f in all_files if f.endswith('.TIF')]
    jpg_files = [os.path.join(path,mission,folder, f) for f in all_files if f.endswith('.JPG')]
    # Adjust the file extension checks to be case-insensitive and ensure path and folder are correct
    other_files = [os.path.join(path,mission, folder, f) for f in all_files if f.lower().endswith(('.nav', '.obs', '.bin', '.mrk'))]
    for file in tif_files:
        shutil.move(file, os.path.join(multispectral_folder, os.path.basename(file)))
    for file in jpg_files:
        shutil.move(file, os.path.join(rgb_folder, os.path.basename(file)))
    for file in other_files:
        shutil.move(file, os.path.join(georeference_folder, os.path.basename(file)))

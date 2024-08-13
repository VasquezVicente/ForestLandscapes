import os
import shutil


path=r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2015\BCI_50ha_2015_02_11_DR1_m1\pics"

listdirs=os.listdir(path)
#move all contents of the subdir to the parent dir
for subdir in listdirs:
    if os.path.isdir(os.path.join(path,subdir)):
        listfiles=os.listdir(os.path.join(path,subdir))
        for file in listfiles:
            shutil.move(os.path.join(path,subdir,file),os.path.join(path,file))
            print(f"Moved {file} to {path}")


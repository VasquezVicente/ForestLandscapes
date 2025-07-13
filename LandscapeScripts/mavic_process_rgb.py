import os   #archivos del sistema
import pandas as pd   #data frames
import shutil   #copiar y mover archivos
import Metashape #fotgrametria
import glob
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from io import StringIO

column_names = [
    "date", "time", "latitude", "longitude", "height_m", "Q", "ns",
    "sdn_m", "sde_m", "sdu_m", "sdne_m", "sdeu_m", "sdun_m", "age_s", "ratio"
]


missions_dir = r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\ECUADOR_yasuni_2025_05_30_M3E"
folders = [folder for folder in os.listdir(missions_dir) if folder.startswith('DJI') and os.path.isdir(os.path.join(missions_dir, folder))]


all_combined=[]
all_df_pos = []
for mission in folders:
    print("procesing mission: ",mission)
    #get the pos event file 
    event_pos_folder=os.path.join(missions_dir,mission)
    event_pos_file = next((os.path.join(event_pos_folder, f) for f in os.listdir(event_pos_folder) if f.endswith("_events.pos")), None)
    with open(event_pos_file, 'r') as f:
        lines = f.readlines()
    start_idx = max(i for i, line in enumerate(lines) if line.startswith('%')) + 1
    data_str = ''.join(lines[start_idx:])
    data_io = StringIO(data_str)
    df_pos = pd.read_csv(data_io, delim_whitespace=True, names=column_names) 
    image_files = sorted([f for f in os.listdir(os.path.join(missions_dir,mission)) if f.lower().endswith(".jpg")])
    photos_rgb_nth = [os.path.join(missions_dir, mission, f) for f in os.listdir(os.path.join(missions_dir, mission)) if f.lower().endswith(".jpg")]
    df_pos['filename']= image_files
    all_combined.extend(photos_rgb_nth)
    all_df_pos.append(df_pos)

df_all_pos = pd.concat(all_df_pos, ignore_index=True)


doc = Metashape.Document()
chunk= doc.addChunk()
dest=os.path.join(missions_dir,"ECUADOR_yasuni_2025_05_30_M3E.psx")
doc.save(dest)
chunk.addPhotos(all_combined)

chunk.exportReference(os.path.join(missions_dir,'original_reference.txt'),delimiter=',')

original_reference = pd.read_csv(os.path.join(missions_dir,"original_reference.txt"), delimiter=',', skiprows=1)


merged_df = pd.merge(original_reference, df_all_pos[['filename', 'longitude', 'latitude', 'height_m']], left_on='#Label',right_on='filename', how='left')

merged_df['X/Longitude'] = merged_df['longitude']
merged_df['Y/Latitude'] = merged_df['latitude']
merged_df['Z/Altitude'] = merged_df['height_m']

merged_df.drop(columns=['filename', 'longitude', 'latitude', 'height_m'], inplace=True)


        #reorder columns to match the original reference
merged_df = merged_df[['#Label', 'X/Longitude', 'Y/Latitude', 'Z/Altitude', 'Yaw', 'Pitch',
            'Roll', 'Error_(m)', 'X_error', 'Y_error', 'Z_error', 'Error_(deg)',
            'Yaw_error', 'Pitch_error', 'Roll_error', 'X_est', 'Y_est', 'Z_est',
            'Yaw_est', 'Pitch_est', 'Roll_est']]
merged_df.to_csv(os.path.join(missions_dir,"merged_reference.txt"), index=False)

doc = Metashape.Document()
doc.open(r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\ECUADOR_yasuni_2025_05_30_M3E\ECUADOR_yasuni_2025_05_30_M3E.psx")
chunk=doc.chunk
len(chunk.cameras)

#convert to utm 18s
out_crs = Metashape.CoordinateSystem("EPSG::32718")
for camera in chunk.cameras:
    if camera.reference.location:
        camera.reference.location = Metashape.CoordinateSystem.transform(camera.reference.location, chunk.crs, out_crs)
chunk.crs = out_crs
chunk.updateTransform()
doc.save()
chunk.matchPhotos(downscale=0,keypoint_limit = 40000, tiepoint_limit = 4000, generic_preselection = True, reference_preselection = True)
doc.save()
chunk.alignCameras(adaptive_fitting=True)
doc.save()

        #build depth maps
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

chunk.calibrateReflectance(use_sun_sensor=True)
        #create an orthoprojection for export
proj = Metashape.OrthoProjection()
proj.crs=Metashape.CoordinateSystem("EPSG::32718")
doc.save()

compression = Metashape.ImageCompression()
compression.tiff_big = True

        #chunk calibrate reflectance

chunk.exportReport(os.path.join(missions_dir,"ECUADOR_yasuni_2025_05_30_report.pdf"))

if chunk.elevation:
    chunk.exportRaster(os.path.join(missions_dir,"ECUADOR_yasuni_2025_05_30_dsm.tif"), source_data = Metashape.ElevationData,projection= proj)
if chunk.orthomosaic:
    chunk.exportRaster(os.path.join(missions_dir,"ECUADOR_yasuni_2025_05_30_orthomosaic.tif"), source_data = Metashape.OrthomosaicData,projection= proj,image_compression = compression)
                                    

print('Processing finished')

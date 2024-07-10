import os   #archivos del sistema
import pandas as pd   #data frames
import shutil   #copiar y mover archivos
import Metashape #fotgrametria

#functions
def create_combined(photos_rgb):
    combined = []
    for i in range(len(photos_rgb)):
        combined.extend([
            photos_rgb[i],
            photos_rgb[i].replace("RGB", "Multispectral").replace("_D.JPG","_MS_G.TIF"),
            photos_rgb[i].replace("RGB", "Multispectral").replace("_D.JPG","_MS_R.TIF"),
            photos_rgb[i].replace("RGB", "Multispectral").replace("_D.JPG","_MS_NIR.TIF"),
            photos_rgb[i].replace("RGB", "Multispectral").replace("_D.JPG","_MS_RE.TIF")
        ])
    return combined


#mavic flight main folders
images_dir=r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\BCI_p14_2024_06_18_M3E"
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



#metashape original images
dest=os.path.join(images_dir.replace('Raw','Products'),"Project",mission+"_medium.psx") #proyecto de metashape en la carpeta de productos
dest2=os.path.join(images_dir.replace('Raw','Products'),"Project")
os.makedirs(dest2, exist_ok=True)

doc = Metashape.Document()
chunk= doc.addChunk()

#add photos to the chunk
flights=os.listdir(images_dir)
photos_rgb1=[os.path.join(images_dir, flights[0], 'RGB', f) for f in os.listdir(os.path.join(images_dir, flights[0], 'RGB'))]
photos_rgb2=[os.path.join(images_dir, flights[1], 'RGB', f) for f in os.listdir(os.path.join(images_dir, flights[1], 'RGB'))]
photos_rgb3=[os.path.join(images_dir, flights[2], 'RGB', f) for f in os.listdir(os.path.join(images_dir, flights[2], 'RGB'))]
chunk1_combined = create_combined(photos_rgb1)
chunk2_combined = create_combined(photos_rgb2)
chunk3_combined = create_combined(photos_rgb3)

all_combined = chunk1_combined + chunk2_combined + chunk3_combined
all_combined= chunk1_combined

chunk.addPhotos(all_combined, filegroups=[5]*(len(photos_rgb1)+len(photos_rgb2)+len(photos_rgb3)), layout=Metashape.MultiplaneLayout)
chunk.addPhotos(all_combined, filegroups=[5]*(len(photos_rgb1)), layout=Metashape.MultiplaneLayout)
chunk.exportReference(os.path.join(images_dir,"Georeference", 'original_reference.txt'),delimiter=',')


#temporal project for the ppk correction, corregir para posible cambio de nombre
doc2 = Metashape.Document()
chunk2= doc2.addChunk()
folders[0]
flight1_dir = os.path.join(images_dir, folders[0],"Georeference/tagged_RGB")
flight2_dir = os.path.join(images_dir, r"DJI_202403061042_002_BCI-AVA\Georeference\tagged_RGB")
flight3_dir = os.path.join(images_dir, r"DJI_202403061107_001_BCI-AVA\Georeference\tagged_RGB")

flight1 = [os.path.join(flight1_dir, f) for f in os.listdir(flight1_dir)]
flight2 = [os.path.join(flight2_dir, f) for f in os.listdir(flight2_dir)]
flight3 = [os.path.join(flight3_dir, f) for f in os.listdir(flight3_dir)]
images_ppk = flight1 + flight2 + flight3
chunk2.addPhotos(images_ppk)
chunk2.exportReference(os.path.join(images_dir,"Georeference",'ppk_reference.txt'),delimiter=',')


# Read in both comma delimited files
ppk_reference = pd.read_csv(os.path.join(images_dir,"Georeference","ppk_reference.txt"), delimiter=',', skiprows=1)
original_reference = pd.read_csv(os.path.join(images_dir,"Georeference","original_reference.txt"), delimiter=',', skiprows=1)
merged_df = pd.merge(original_reference, ppk_reference[['#Label', 'X/Longitude', 'Y/Latitude', 'Z/Altitude']], on='#Label', how='left')

merged_df['X/Longitude'] = merged_df['X/Longitude_y']
merged_df['Y/Latitude'] = merged_df['Y/Latitude_y']
merged_df['Z/Altitude'] = merged_df['Z/Altitude_y']

merged_df.drop(columns=['X/Longitude_x', 'X/Longitude_y', 'Y/Latitude_x', 'Y/Latitude_y', 'Z/Altitude_x', 'Z/Altitude_y'], inplace=True)
#reorder columns to match the original reference
merged_df = merged_df[['#Label', 'X/Longitude', 'Y/Latitude', 'Z/Altitude', 'Yaw', 'Pitch',
       'Roll', 'Error_(m)', 'X_error', 'Y_error', 'Z_error', 'Error_(deg)',
       'Yaw_error', 'Pitch_error', 'Roll_error', 'X_est', 'Y_est', 'Z_est',
       'Yaw_est', 'Pitch_est', 'Roll_est']]
merged_df.to_csv(os.path.join(images_dir,"Georeference","merged_reference.txt"), index=False)


#BACK TO PROCESSING
#open document
#delete the original file 
doc = Metashape.Document()
doc.save(dest)
chunk= doc.addChunk()
chunk.addPhotos(all_combined, filegroups=[5]*(len(photos_rgb1)+len(photos_rgb2)+len(photos_rgb3)), layout=Metashape.MultiplaneLayout)#tres vuelos
chunk.addPhotos(all_combined, filegroups=[5]*(len(photos_rgb1)), layout=Metashape.MultiplaneLayout)#un vuelo
chunk.importReference(os.path.join(images_dir,"Georeference","merged_reference.txt"),delimiter=',',columns="nxyzXYZabcABC|[]")
doc.save()


#convert to utm 17N
out_crs = Metashape.CoordinateSystem("EPSG::32617")
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
chunk.buildDepthMaps(downscale = 4, filter_mode = Metashape.AggressiveFiltering)
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
proj.crs=Metashape.CoordinateSystem("EPSG::32617")
doc.save()

#chunk calibrate reflectance

chunk.exportReport(dest.replace('medium','report').replace('.psx','.pdf'))

out_dir= images_dir.replace('Raw','Products')
folder_cloud= os.path.join(out_dir, 'Cloudpoint')
os.makedirs(folder_cloud, exist_ok=True)
folder_dsm= os.path.join(out_dir, 'DSM')
os.makedirs(folder_dsm, exist_ok=True)
folder_orthomosaic= os.path.join(out_dir, 'Orthophoto')
os.makedirs(folder_orthomosaic, exist_ok=True)
mission

if chunk.point_cloud:
            chunk.exportPointCloud(os.path.join(folder_cloud,mission.replace("M3E","cloud.las")), source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatLAS, crs = Metashape.CoordinateSystem("EPSG::32617"))
            chunk.exportPointCloud(os.path.join(folder_cloud,mission.replace("M3E","cloud.ply")), source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatPLY, crs = Metashape.CoordinateSystem("EPSG::32617"))
if chunk.elevation:
            chunk.exportRaster(os.path.join(folder_dsm,mission.replace("M3E","dsm.tif")), source_data = Metashape.ElevationData,projection= proj)
if chunk.orthomosaic:
            chunk.exportRaster(os.path.join(folder_orthomosaic,mission.replace("M3E","orthomosaic.tif")), source_data = Metashape.OrthomosaicData,projection= proj)

print('Processing finished')



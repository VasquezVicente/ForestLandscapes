import os
import pandas as pd
import Metashape
import re
import time

def process_drone_images(images_dir):
      start_time = time.time()
      doc = Metashape.Document()
      dest = images_dir
      if 'P4P' in images_dir:
            project_name=os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("P4P", "medium")+'.psx')
            dest= images_dir.replace("LandscapeRaw","LandscapeProducts").replace("Images",os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("P4P", "medium")+'.psx'))
      elif 'EBEE' in images_dir:
            project_name=os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("EBEE", "medium")+'.psx')
            dest= images_dir.replace("LandscapeRaw","LandscapeProducts").replace("Images",os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("EBEE", "medium")+'.psx'))
      elif 'INSPIRE' in images_dir:
            project_name=os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("INSPIRE", "medium")+'.psx')
            dest= images_dir.replace("LandscapeRaw","LandscapeProducts").replace("Images",os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("INSPIRE", "medium")+'.psx'))
      elif 'SOLO' in images_dir:
            project_name=os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("SOLO", "medium")+'.psx')
            dest= images_dir.replace("LandscapeRaw","LandscapeProducts").replace("Images",os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("SOLO", "medium")+'.psx'))
      doc.save(dest)
      chunk = doc.addChunk()
      photos = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
      chunk.addPhotos(photos)
      doc.save()
      out_crs = Metashape.CoordinateSystem("EPSG::32617")
      for camera in chunk.cameras:
            if camera.reference.location:
                camera.reference.location = Metashape.CoordinateSystem.transform(camera.reference.location, chunk.crs, out_crs)
      chunk.crs = out_crs
      chunk.updateTransform()
      doc.save()

#align photos
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

#create an orthoprojection for export
      proj = Metashape.OrthoProjection()
      proj.crs=Metashape.CoordinateSystem("EPSG::32617")
      doc.save()

      chunk.exportReport(dest.replace('medium','report').replace('.psx','.pdf'))

      if chunk.point_cloud:
            chunk.exportPointCloud(dest.replace("Project","Cloudpoint").replace("medium", "cloud").replace('.psx','.las'), source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatLAS, crs = Metashape.CoordinateSystem("EPSG::32617"))
            chunk.exportPointCloud(dest.replace("Project","Cloudpoint").replace("medium", "cloud").replace('.psx','.ply'), source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatPLY, crs = Metashape.CoordinateSystem("EPSG::32617"))
      if chunk.elevation:
            chunk.exportRaster(dest.replace("Project","DSM").replace("medium", "dsm").replace('.psx','.tif'), source_data = Metashape.ElevationData,projection= proj)
      if chunk.orthomosaic:
            chunk.exportRaster(dest.replace("Project","Orthophoto").replace("medium", "orthomosaic").replace('.psx','.tif'), source_data = Metashape.OrthomosaicData,projection= proj)

      print('Processing finished')

      end_time = time.time()
      elapsed_time = end_time - start_time
      print(f'Total processing time: {elapsed_time} seconds')


server= r'\\stri-sm01\ForestLandscapes'
#read paths
paths= pd.read_csv(os.path.join(server, 'filepaths.csv'),low_memory=False)
year_to_check= input('Enter year to check: ')


#filter paths by type == LanscapeRaw and year == year_to_check
filtered_paths= paths[(paths['type']=='LandscapeRaw') & (paths['year']==year_to_check) & (paths['source']=='Drone')]

#filtered paths unique mission names
mission_names= filtered_paths['mission'].unique()

for index, mission in enumerate(mission_names):
    print(f"{index}: {mission}")

dates = input("Please select dates for the missions (separated by spaces): ")
selected_dates = [int(date) for date in dates.split()]

for i in selected_dates:
    if i < len(mission_names):
        print("processing", mission_names[i])
        process_drone_images(os.path.join(server, "LandscapeRaw", "Drone", str(year_to_check), mission_names[i], "Images"))
    else:
        print(f"Invalid index: {i}")


import os
import pandas as pd
import Metashape
import re

def process_drone_images(images_dir):
      doc = Metashape.Document()
      project_name=os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("P4P", "medium")+'.psx')
      dest= images_dir.replace("LandscapeRaw","LandscapeProducts").replace("Images",os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("P4P", "medium")+'.psx'))
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

      chunk.exportReport(dest.replace(project_name,project_name.replace("medium.psx", "medium.pdf")))

      if chunk.point_cloud:
            chunk.exportPointCloud(dest.replace("Project","Cloudpoint").replace("medium.psx", "cloud.las"), source_data = Metashape.PointCloudData)
            chunk.exportPointCloud(dest.replace("Project","Cloudpoint").replace("medium.psx", "cloud.ply"), source_data = Metashape.PointCloudData)
      if chunk.elevation:
            chunk.exportRaster(dest.replace("Project","DSM").replace("medium.psx", "dsm.tif"), source_data = Metashape.ElevationData,projection= proj)
      if chunk.orthomosaic:
            chunk.exportRaster(dest.replace("Project","Orthophoto").replace("medium.psx", "orthomosaic.tif"), source_data = Metashape.OrthomosaicData,projection= proj)

      print('Processing finished')


server= r'\\stri-sm01\ForestLandscapes'
#read paths
paths= pd.read_csv(os.path.join(server, 'filepaths.csv'))
year_to_check= input('Enter year to check: ')


#filter paths by type == LanscapeRaw and year == year_to_check
filtered_paths= paths[(paths['type']=='LandscapeRaw') & (paths['year']==year_to_check) & (paths['source']=='Drone')]

#filtered paths unique mission names
mission_names= filtered_paths['mission'].unique()


for mission in mission_names:
    if not os.path.exists(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission)):
        print('Mission name does not exist in Prodcuts folder, creating it!')
        os.makedirs(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission))
        os.makedirs(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'Orthophoto'))
        os.makedirs(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'DSM'))
        os.makedirs(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'Cloudpoint'))
        os.makedirs(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'Project'))

#lets do orthomosaics first and see
for mission in mission_names[21:30]:
    if not os.listdir(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'Orthophoto')):
        print("Directory Orthophoto is empty")
        boolean_ortho= False
    else:
        print("Directory Orthophoto is not empty")
        boolean_ortho= True
    if not os.listdir(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'DSM')):
        print("Directory DSM is empty")
        boolean_dsm= False
    else: 
        print("Directory DSM is not empty")
        boolean_dsm= True
    if not os.listdir(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'Cloudpoint')):
        print("Directory Cloudpoint is empty")
        boolean_cloud= False
    else:
        print("Directory Cloudpoint is not empty")
        boolean_cloud= True
    if not os.listdir(os.path.join(server,"LandscapeProducts","Drone",year_to_check,mission,'Project')):
        print("Directory Project is empty")
        boolean_project= False
    if boolean_cloud == False and boolean_dsm == False and boolean_ortho == False and boolean_project == False:
        print('Missions hasnt been processed', mission)
        #process the mission
        process_drone_images(os.path.join(server,"LandscapeRaw","Drone",year_to_check,mission,"Images"))


for mission in mission_names[23:30]:
     process_drone_images(os.path.join(server,"LandscapeRaw","Drone",year_to_check,mission,"Images"))


for mission in mission_names[31:50]:
     process_drone_images(os.path.join(server,"LandscapeRaw","Drone",year_to_check,mission,"Images"))
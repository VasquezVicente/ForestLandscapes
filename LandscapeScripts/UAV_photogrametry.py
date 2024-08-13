import os
import pandas as pd
import Metashape
import re
import time

#for image folders that have a set of images selected. all images within folder must be good and usables , this means no extra images
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

def reexport_cloud(project_dir):
    # Load the project
    doc = Metashape.Document()
    doc.open(project_dir)
    chunk = doc.chunk

    mission= os.path.basename(project_dir)
    mission= mission.split("_")
    last_bit= "_".join(mission[5:])
    quality= last_bit.split("_")[0].split(".")[0]
    last_bit1= last_bit.replace(quality,"cloud").replace(".psx", ".las")
    mission= "_".join(mission[0:5])   
    mission = mission+"_"+ last_bit1

    cloud_path= os.path.join(os.path.dirname(project_dir), mission).replace("Project", "Cloudpoint")
    # Export the dense point cloud
    proj = Metashape.OrthoProjection()
    proj.crs=Metashape.CoordinateSystem("EPSG::32617")
    print(cloud_path)
    chunk.exportPointCloud(cloud_path, source_data=Metashape.PointCloudData, format=Metashape.PointCloudFormatLAS, crs=Metashape.CoordinateSystem("EPSG::32617"))

def process_project(project_dir):
    doc= Metashape.Document()
    doc.open(project_dir, read_only=False)
    chunk = doc.chunk
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

    chunk.exportReport(project_dir.replace('medium','report').replace('.psx','.pdf'))

    if chunk.point_cloud:
            chunk.exportPointCloud(project_dir.replace("Project","Cloudpoint").replace("medium", "cloud").replace('.psx','.las'), source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatLAS, crs = Metashape.CoordinateSystem("EPSG::32617"))
            chunk.exportPointCloud(project_dir.replace("Project","Cloudpoint").replace("medium", "cloud").replace('.psx','.ply'), source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatPLY, crs = Metashape.CoordinateSystem("EPSG::32617"))
    if chunk.elevation:
            chunk.exportRaster(project_dir.replace("Project","DSM").replace("medium", "dsm").replace('.psx','.tif'), source_data = Metashape.ElevationData,projection= proj)
    if chunk.orthomosaic:
            chunk.exportRaster(project_dir.replace("Project","Orthophoto").replace("medium", "orthomosaic").replace('.psx','.tif'), source_data = Metashape.OrthomosaicData,projection= proj)

    print('Processing finished')

def reexport_ortho(project_dir):
    doc = Metashape.Document()
    doc.open(project_dir)
    chunk = doc.chunk
    mission= os.path.basename(project_dir)
    mission= mission.split("_")
    last_bit= "_".join(mission[5:])
    quality= last_bit.split("_")[0].split(".")[0]
    last_bit1= last_bit.replace(quality,"orthomosaic").replace(".psx", ".tif")
    mission= "_".join(mission[0:5])   
    mission = mission+"_"+ last_bit1
    compression = Metashape.ImageCompression()
    compression.tiff_big = True
    cloud_path= os.path.join(os.path.dirname(project_dir), mission).replace("Project", "Orthophoto")
    # Export the dense point cloud
    proj = Metashape.OrthoProjection()
    proj.crs=Metashape.CoordinateSystem("EPSG::32617")
    print(cloud_path)
    chunk.exportRaster(cloud_path, source_data = Metashape.OrthomosaicData,projection= proj,image_compression = compression)

def export_products(project_dir):
      project_dir=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2013\BCI_50ha_2013_06_11_DR1_m1\Project\BCI_50ha_2013_06_11_medium.psx"
      doc = Metashape.Document()
      doc.open(project_dir)
      chunk = doc.chunk

      mission= os.path.basename(project_dir)
      parts= mission.split("_")
      mission= "_".join(parts[0:5])

      last_bit= "_".join(parts[5:])

      quality= last_bit.split("_")[0].split(".")[0]

      ortho= last_bit.replace(quality,"orthomosaic").replace(".psx", ".tif")
      dsm= last_bit.replace(quality,"dsm").replace(".psx", ".tif")
      cloud= last_bit.replace(quality,"cloud").replace(".psx", ".las")
      
      cloud_path= os.path.join(os.path.dirname(project_dir), mission+"_"+cloud).replace("Project", "Cloudpoint")
      ortho_path= os.path.join(os.path.dirname(project_dir), mission+"_"+ortho).replace("Project", "Orthophoto")
      dsm_path= os.path.join(os.path.dirname(project_dir), mission+"_"+dsm).replace("Project", "DSM")

      cloud_dir= os.path.dirname(project_dir).replace("Project", "Cloudpoint")
      ortho_dir= os.path.dirname(project_dir).replace("Project", "Orthophoto")
      dsm_dir= os.path.dirname(project_dir).replace("Project", "DSM")
      os.makedirs(cloud_dir, exist_ok=True)
      os.makedirs(ortho_dir, exist_ok=True)
      os.makedirs(dsm_dir, exist_ok=True)

      proj = Metashape.OrthoProjection()
      proj.crs=Metashape.CoordinateSystem("EPSG::32617")
      compression = Metashape.ImageCompression()
      compression.tiff_big = True

      chunk.exportReport(project_dir.replace('medium','report').replace('.psx','.pdf'))

      if chunk.point_cloud:
            chunk.exportPointCloud(cloud_path, source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatLAS, crs = Metashape.CoordinateSystem("EPSG::32617"))
            chunk.exportPointCloud(cloud_path.replace(".las",".ply"), source_data = Metashape.PointCloudData,format = Metashape.PointCloudFormatPLY, crs = Metashape.CoordinateSystem("EPSG::32617"))
      if chunk.elevation:
            chunk.exportRaster(dsm_path, source_data = Metashape.ElevationData,projection= proj)
      if chunk.orthomosaic:
            chunk.exportRaster(ortho_path, source_data = Metashape.OrthomosaicData,projection= proj,image_compression = compression)

      print('Processing finished')

process_project(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2016\BCI_50ha_2016_04_06_SOLO\Project\BCI_50ha_2016_04_06_medium.psx")
process_project(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2016\BCI_25ha_2016_12_23_EBEE2\Project\BCI_25ha_2016_12_23_medium.psx")

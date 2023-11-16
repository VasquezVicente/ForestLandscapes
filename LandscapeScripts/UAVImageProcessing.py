import Metashape
import os, sys, time
import re

server_dir= r'\\stri-sm01\ForestLandscapes'

#inputs that i need, the raw is static. i need the year, the mission, the setting. normally medium for 50ha
#in that case i need downscale_matchphotos==0 as higuest, downscales_depthmaps==4 that is medium, and filter_mode==Metashape.AggressiveFiltering


# Checking compatibility
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != "2.0":
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))
else:
    print("Metshape app is compatible, YAY!")

target= os.path.join(server_dir,"LandscapeRaw","Drone","2021","BCI_50ha_2021_09_29_P4P","Images")

photos= []
for file_name in os.listdir(target):
    photo = os.path.join(target, file_name)
    print(photos)
    photos.append(photo)



#add photos and create project

def process_drone_images(images_dir):
      doc = Metashape.Document()
      project_name=os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("P4P", "medium")+'.psx')
      dest= images_dir.replace("LandscapeRaw","LandscapeProducts").replace("Images",os.path.join('Project',re.split(r"[\\/]", images_dir)[7].replace("P4P", "medium")+'.psx'))
      doc.save(dest)
      chunk = doc.addChunk()
      photos= os.listdir(images_dir,full_path=True)
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
            chunk.exportRaster(dest.replace("Project","Cloudpoint").replace("medium.psx", "orthomosaic.tif"), source_data = Metashape.OrthomosaicData,projection= proj)

      print('Processing finished')


























filepaths= pd.read_csv(os.path.join(server_dir, "filepaths.csv"),low_memory=False)

# filter filepaths to only include raw images, drone and 2021
filtered_data= filepaths[(filepaths['type']=='LandscapeRaw') & 
                         (filepaths['source']=='Drone') & 
                         (filepaths['year']=='2021')&
                         (filepaths['mission']=='BCI_50ha_2021_09_29_P4P')]

#join columns to create the full path
filtered_data['full_path']=pd.concat(filtered_data['server'],
                                            filtered_data['partition'],
                                            filtered_data['type'],
                                            filtered_data['source'],
                                            filtered_data['year'],
                                            filtered_data['mission'],
                                            filtered_data['file'])
filtered_data['full_path'] = filtered_data.apply(lambda x: ''.join(x), axis=1)
filtered_data.dtypes
filepaths = pd.read_csv(os.path.join(server_dir, "filepaths.csv"))
year= input("Enter the year of the flight: ")


def drone_processing(image_folder,output_folder,site,new_folder_name):
    #add the images
    

filepaths.columns
copy

def drone_processing1(image_folder,output_folder,site,new_folder_name):
    if not os.path.exists(image_folder):
        print('image folders does not exist, check for folder or check that the slashes arent backwards')
    if not os.path.exists(output_folder):
        print('image folders does not exist, check for folder or check that the slashes arent backwards')
    if site=="BCI_50ha":
            photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])

            doc = Metashape.Document()
            doc.save(output_folder +'/'+re.split(r"[\\/]", image_folder)[4]+ '.psx')
            chunk = doc.addChunk()

            chunk.addPhotos(photos)
            doc.save()

            print(str(len(chunk.cameras)) + " images loaded")
            if not len(chunk.cameras)==304:
                print("not the expected amount of picture, check the image folder manually")

            chunk.matchPhotos(downscale=0,keypoint_limit = 40000, tiepoint_limit = 4000, generic_preselection = True, reference_preselection = True)
            doc.save()

            chunk.alignCameras(adaptive_fitting=True)
            doc.save()

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

            proj = Metashape.OrthoProjection()
            proj.crs=Metashape.CoordinateSystem("EPSG::32617")
            doc.save()
            # export results
            chunk.exportReport(output_folder + re.split(r"[\\/]", image_folder)[4]+ '.pdf')

            if chunk.point_cloud:
                chunk.exportPointCloud(output_folder + '/Cloudpoint/'+re.split(r"[\\/]", image_folder)[4]+ '.las', source_data = Metashape.PointCloudData)
            if chunk.elevation:
                chunk.exportRaster(output_folder + '/DEM/'+re.split(r"[\\/]", image_folder)[4]+ '.tif', source_data = Metashape.ElevationData,projection= proj)
            if chunk.orthomosaic:
                chunk.exportRaster(output_folder + '/Orthophoto/'+re.split(r"[\\/]", image_folder)[4]+'.tif', source_data = Metashape.OrthomosaicData,projection= proj)

            print('Processing finished, results saved to ' + output_folder + '.')
    elif site=="BCI_ava":
            photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])
            doc = Metashape.Document()
            doc.save(output_folder +'/'+new_folder_name+'/Project/'+new_folder_name.replace("P4P", "medium")+ '.psx')
            chunk = doc.addChunk()
            chunk.addPhotos(photos)
            doc.save()
            out_crs=Metashape.CoordinateSystem("EPSG::32617")
            for camera in chunk.cameras:
                if camera.reference.location:
                    camera.reference.location = Metashape.CoordinateSystem.transform(camera.reference.location, chunk.crs, out_crs)
            chunk.crs=Metashape.CoordinateSystem("EPSG::32617")
            chunk.crs
            doc.save()
            chunk.matchPhotos(downscale=0,keypoint_limit = 40000, tiepoint_limit = 4000, generic_preselection = True, reference_preselection = True)
            doc.save()
            chunk.alignCameras()
            doc.save()
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
            proj = Metashape.OrthoProjection()
            proj.crs=Metashape.CoordinateSystem("EPSG::32617")
            doc.save()
            chunk.exportReport(output_folder +'/'+new_folder_name+'/Project/'+new_folder_name.replace("P4P", "medium")+ '.pdf')
            if chunk.point_cloud:
                    chunk.exportPointCloud(output_folder +'/'+new_folder_name+ '/Cloudpoint/'+new_folder_name.replace("P4P", "medium")+ '.las', source_data = Metashape.PointCloudData)
            if chunk.elevation:
                    chunk.exportRaster(output_folder +'/'+ new_folder_name+'/DEM/'+new_folder_name.replace("P4P", "medium")+ '.tif', source_data = Metashape.ElevationData,projection= proj)
            if chunk.orthomosaic:
                    chunk.exportRaster(output_folder +'/'+new_folder_name+ '/Orthophoto/'+new_folder_name.replace("P4P", "medium")+'.tif', source_data = Metashape.OrthomosaicData,projection= proj)
            print('Processing finished, results saved to ' + output_folder + '.')
    elif site=="BCI_whole":
            photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])
            doc = Metashape.Document()
            doc.save(output_folder +'/'+new_folder_name+'/Project/'+new_folder_name.replace("P4P", "medium")+ '.psx')
            chunk = doc.addChunk()
            chunk.addPhotos(photos)
            doc.save()
            out_crs=Metashape.CoordinateSystem("EPSG::32617")
            for camera in chunk.cameras:
                if camera.reference.location:
                    camera.reference.location = Metashape.CoordinateSystem.transform(camera.reference.location, chunk.crs, out_crs)
            chunk.crs=Metashape.CoordinateSystem("EPSG::32617")
            chunk.crs
            doc.save()
            chunk.matchPhotos(downscale=0,keypoint_limit = 40000, tiepoint_limit = 4000, generic_preselection = True, reference_preselection = True)
            doc.save()
            chunk.alignCameras()
            doc.save()
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
            proj = Metashape.OrthoProjection()
            proj.crs=Metashape.CoordinateSystem("EPSG::32617")
            doc.save()
            chunk.exportReport(output_folder +'/'+new_folder_name+'/Project/'+new_folder_name.replace("EBEE", "medium")+ '.pdf')
            if chunk.point_cloud:
                    chunk.exportPointCloud(output_folder +'/'+new_folder_name+ '/Cloudpoint/'+new_folder_name.replace("EBEE", "medium")+ '.las', source_data = Metashape.PointCloudData)
            if chunk.elevation:
                    chunk.exportRaster(output_folder +'/'+ new_folder_name+'/DEM/'+new_folder_name.replace("EBEE", "medium")+ '.tif', source_data = Metashape.ElevationData,projection= proj)
            if chunk.orthomosaic:
                    chunk.exportRaster(output_folder +'/'+new_folder_name+ '/Orthophoto/'+new_folder_name.replace("EBEE", "medium")+'.tif', source_data = Metashape.OrthomosaicData,projection= proj)
            print('Processing finished, results saved to ' + output_folder + '.')


drone_processing(r"C:\Users\VasquezV\repo\temp_processing\BCI_50ha_2023_07_11_P4P\Images",
                 r"C:\Users\VasquezV\repo\temp_processing\BCI_50ha_2023_07_11_P4P",
                 "BCI_ava",
                 "BCI_50ha_2023_07_11_P4P")

from scripts.TreeCrownKit import crop_tif_group
dir=r"D:/10hawork"
crop_tif_group(dir,"box", [625782,1012239,626805,1012381])


import Metashape
import os, sys, time
import pandas as pd
from tqdm import tqdm
import re

server_dir= r'\\stri-sm01\ForestLandscapes'

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))
else:
    print("Metshape app is compatible, YAY!")
    
filepaths = pd.read_csv(os.path.join(server_dir, "filepaths.csv"))
year= input("Enter the year of the flight: ")


def drone_processing(image_folder,output_folder,site,new_folder_name):
    #add the images
    


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


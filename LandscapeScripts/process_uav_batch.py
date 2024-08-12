import Metashape
import os
import time


def process_project(project_dir):
    doc= Metashape.Document()
    doc.open(project_dir)
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

#test1
process_project(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2021\GAMBOA_p09_2021_09_24_P4P\Project\GAMBOA_p09_2021_09_24_medium.psx")
process_project(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2015\AGUASALUD_eastarea_2015_04_16_DR1\Project\AGUASALUD_eastarea_2015_04_16_medium.psx")
process_project(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2015\AGUASALUD_northarea_2015_04_24_DR1\Project\AGUASALUD_northarea_2015_04_24_medium.psx")
process_project(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2015\AGUASALUD_southarea_2015_04_19_DR1\Project\AGUASALUD_southarea_2015_04_19_medium.psx")
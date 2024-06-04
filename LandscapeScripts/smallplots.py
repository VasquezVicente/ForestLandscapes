import os
import pandas as pd
import Metashape
import re
import time
import numpy as np

#check if all product missions have an ortho
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

server_dir = r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone"
#check for mission products
years = [2018,2019,2020,2021,2022,2023]
table_data = []
for year in years:
    print(f'Processing year {year}')
    missions_product = [f for f in os.listdir(os.path.join(server_dir, str(year))) if os.path.isdir(os.path.join(server_dir, str(year), f))]
    for mission in missions_product:
        ortho_folder = os.path.join(server_dir.replace('Raw', 'Products'), str(year), mission, 'Orthophoto')
        dsm_folder = os.path.join(server_dir.replace('Raw', 'Products'), str(year), mission, 'DSM')
        cloud_folder = os.path.join(server_dir.replace('Raw', 'Products'), str(year), mission, 'Cloudpoint')
        list_ortho = [f for f in os.listdir(ortho_folder) if f.endswith('.tif')]
        list_dsm = [f for f in os.listdir(dsm_folder) if f.endswith('.tif')]
        list_cloud = [f for f in os.listdir(cloud_folder) if f.endswith('.las') or f.endswith('.ply')]
        
        # Concatenate the file names with a comma separator
        ortho = ', '.join(list_ortho) if list_ortho else np.nan
        dsm = ', '.join(list_dsm) if list_dsm else np.nan
        cloud = ', '.join(list_cloud) if list_cloud else np.nan
        
        table_data.append({
            'Year': year,
            'Mission Product': mission,
            'Orthomosaic': ortho,
            'DSM': dsm,
            'Cloud': cloud
        })

df = pd.DataFrame(table_data)
for row in df.itertuples():
    print(row)

# Export to csv
df.to_csv(os.path.join(r"\\stri-sm01\ForestLandscapes", 'drone_products.csv'), index=False)
site_counts = df['Mission Product'].apply(lambda x: '_'.join(x.split('_')[0:2])).value_counts()
site_counts_df = site_counts.reset_index()
site_counts_df.columns = ['Site', 'Count']


#process drone images that dont have ortho
for mission in df['Mission Product']:
    try:
        if df.loc[df['Mission Product'] == mission, 'Orthomosaic'].isnull().values[0]:
            print(f'Processing {mission}')
            images_dir = os.path.join(server_dir, str(df.loc[df['Mission Product'] == mission, 'Year'].values[0]), mission, 'Images')
            
            # Check if the directory contains any .jpg files
            if any(fname.lower().endswith(('.jpg', '.jpeg')) for fname in os.listdir(images_dir)):
                process_drone_images(images_dir)
                print(f'Finished processing {mission}')
    except Exception as e:
        print(f'Error processing {mission}: {e}')

import os
import shutil
import pandas as pd
import time
import rasterio
import laspy
# we are going to scout the LandscapeProducts directory looking for unique sits
path= r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone"

years=[2024,2023,2022,2021,2020,2019,2018,2017,2016,2015]
years= [str(year) for year in years]
paths= []
for year in years:
    path1= os.path.join(path, year)
    paths.append(path1)

#list all the dirs in all the paths
missions= []
for path in paths:
    dirs= os.listdir(path)
    for dir in dirs:
        missions.append(dir)

missions_df= pd.DataFrame(missions, columns=['mission'])


missions_df["site"]= missions_df["mission"].str.split("_").str[0]
missions_df["plot"]= missions_df["mission"].str.split("_").str[1]
missions_df["year"]= missions_df["mission"].str.split("_").str[2]
missions_df["month"]= missions_df["mission"].str.split("_").str[3]
missions_df["day"]= missions_df["mission"].str.split("_").str[4]
missions_df["drone"]= missions_df["mission"].str.split("_").str[5]


plots=['metrop','p12','p14','10ha','25ha','knightlang','p18','p25','p26','p21','p22','p23','p24','10ha','gigante2',
        'gigantefertilizante','p06','p09','p15','75ha','alfajia2','alfajia','che','porton',
          'site16','site3031','site41','site66','site70','metrop2','senderocc','eastarea',
           'northarea','southarea','p16','p32','roubik','p27','p28']
metadata_all= []
for PLOT in plots:
    thisplot=missions_df[missions_df['plot']==PLOT]
    print(thisplot)
    
    ortho_paths = []
    dsm_paths = []
    cloud_paths= []
    for mission in thisplot['mission']:
        parts = mission.split("_")
        year = parts[2]
        drone = parts[5]
        if len(parts) > 6:
            extra = parts[6]
            ortho_path = os.path.join(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone", year, mission, "Orthophoto", f"{mission.replace(f'_{drone}', '_orthomosaic')}.tif")
        else:
            ortho_path = os.path.join(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone", year, mission, "Orthophoto", f"{mission.replace(drone, 'orthomosaic')}.tif")
        ortho_paths.append(ortho_path)
    for mission in thisplot['mission']:
        parts = mission.split("_")
        year = parts[2]
        drone = parts[5]
        if len(parts) > 6:
            extra = parts[6]
            dsm_path = os.path.join(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone", year, mission, "DSM", f"{mission.replace(f'_{drone}', '_dsm')}.tif")
        else:
            dsm_path = os.path.join(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone", year, mission, "DSM", f"{mission.replace(drone, 'dsm')}.tif")
        dsm_paths.append(dsm_path)
    for mission in thisplot['mission']:
        year= mission.split("_")[2]
        drone= mission.split("_")[5]
        if len(mission.split("_"))>6:
            extra= mission.split("_")[6]
            cloud_path= os.path.join(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone",year, mission, "Cloudpoint", f"{mission.replace(f'_{drone}', '_cloud')}.las")
        else:
            cloud_path= os.path.join(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone",year, mission, "Cloudpoint", f"{mission.replace(drone, 'cloud')}.las")
        cloud_paths.append(cloud_path)
    # Initialize an empty list to store metadata
    metadata_list = []

    # Extract metadata for orthomosaics
    for orthomosaic in ortho_paths:
        with rasterio.open(orthomosaic) as src:
            meta = src.meta
            thebounds = src.bounds
            meta["xmin"] = thebounds.left
            meta["xmax"] = thebounds.right
            meta["ymin"] = thebounds.bottom
            meta["ymax"] = thebounds.top
            meta['product'] = os.path.basename(orthomosaic)	
            meta['type'] = 'orthomosaic'
            
            # Extract Affine transform components
            transform = src.transform
            meta['scale_x'] = transform.a
            meta['shear_x'] = transform.b
            meta['translate_x'] = transform.c
            meta['shear_y'] = transform.d
            meta['scale_y'] = transform.e
            meta['translate_y'] = transform.f
            
            # Append the metadata dictionary to the list
            metadata_list.append(meta)

    # Extract metadata for DSMs
    for dsm in dsm_paths:
        with rasterio.open(dsm) as src:
            meta = src.meta
            thebounds = src.bounds
            meta["xmin"] = thebounds.left
            meta["xmax"] = thebounds.right
            meta["ymin"] = thebounds.bottom
            meta["ymax"] = thebounds.top
            meta['product'] = os.path.basename(dsm)
            meta['type'] = 'dsm'
            
            # Extract Affine transform components
            transform = src.transform
            meta['scale_x'] = transform.a
            meta['shear_x'] = transform.b
            meta['translate_x'] = transform.c
            meta['shear_y'] = transform.d
            meta['scale_y'] = transform.e
            meta['translate_y'] = transform.f
            
            # Append the metadata dictionary to the list
            metadata_list.append(meta)
    # Extract meta_clouddata for point clouds

    meta_cloud2=[]
    for cloud in cloud_paths:
        meta_cloud={}
        src=laspy.read(cloud)
        meta_cloud['product'] =os.path.basename(cloud)
        meta_cloud['type'] = 'point_cloud'
        meta_cloud["point_n"]= src.header.point_records_count
        meta_cloud["xmin"]= src.header.x_min
        meta_cloud["xmax"]= src.header.x_max
        meta_cloud["ymin"]= src.header.y_min
        meta_cloud["ymax"]= src.header.y_max
        meta_cloud["zmin"]= src.header.z_min
        meta_cloud["zmax"]= src.header.z_max
        meta_cloud["scale_x"]= src.header.scale[0]
        meta_cloud["scale_y"]= src.header.scale[1]
        meta_cloud["scale_z"]= src.header.scale[2]
        meta_cloud["offset_x"]= src.header.offset[0]
        meta_cloud["offset_y"]= src.header.offset[1]
        meta_cloud["offset_z"]= src.header.offset[2]
        #Attempt to get CRS from GeoAsciiParamsVlr
        geo_ascii_vlr = src.header.vlrs.get('GeoAsciiParamsVlr')
        if geo_ascii_vlr:
            meta_cloud["crs"] = geo_ascii_vlr[0]
        else:
            # Fallback to WktCoordinateSystemVlr if GeoAsciiParamsVlr is not available
            wkt_vlr = next((vlr for vlr in src.header.vlrs if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr)), None)
            meta_cloud["crs"] = wkt_vlr.string if wkt_vlr else None
        meta_cloud["las_version"] = f"{src.header.version.major}.{src.header.version.minor}"
        dimensions = []


        # Iterate through each dimension in src.point_format.dimensions
        for dim in src.point_format.dimensions:
            dimensions.append(dim.name)
        meta_cloud["dimensions"]= ", ".join(dimensions)
        meta_cloud2.append(meta_cloud)

    # Convert the list of metadata dictionaries to a DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    cloud_df= pd.DataFrame(meta_cloud2)
    metadata_df= pd.concat([metadata_df, cloud_df], ignore_index=True)

    print(metadata_df)
    metadata_df["site"]= metadata_df["product"].str.split("_").str[0]
    metadata_df["plot"]= metadata_df["product"].str.split("_").str[1]
    metadata_df["year"]= metadata_df["product"].str.split("_").str[2]
    metadata_df["month"]= metadata_df["product"].str.split("_").str[3]
    metadata_df["day"]= metadata_df["product"].str.split("_").str[4]
    metadata_df["type"]= metadata_df["product"].str.split("_").str[5]
    metadata_df["product_temp"]= metadata_df["product"].str.split(".").str[0]

    thisplot2=thisplot.copy()
    thisplot3=thisplot.copy()
    thisplot["product_temp"] = thisplot.apply(lambda row: row["mission"].replace(row["drone"], "orthomosaic"), axis=1)
    thisplot2["product_temp"]= thisplot2.apply(lambda row: row["mission"].replace(row["drone"], "dsm"), axis=1)
    thisplot3["product_temp"]= thisplot3.apply(lambda row: row["mission"].replace(row["drone"], "point_cloud"), axis=1)

    thisplot= pd.concat([thisplot, thisplot2, thisplot3], ignore_index=True)
    metadata=metadata_df.merge(thisplot[["product_temp","drone"]], left_on='product_temp', right_on='product_temp', how='left')
   
    #save the csv
    metadata= metadata_df.drop(columns=['product_temp'])
    metadata_all.append(metadata)

met= pd.concat(metadata_all, ignore_index=True)
met.to_csv(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\metadata_all.csv", index=False)


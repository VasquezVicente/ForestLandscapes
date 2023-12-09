# ForestLanscapes
https://doi.org/10.5281/zenodo.10328609
## The repository encompasess all the code necessary to replicate the products in the following publications: 
    - "Time series of the Barro Colorado Island 50ha plot photogrametry orthomosaics and digital surface model captured with DJI Phantom 4 Pro drone: Globally aligned and vertically corrected."
    - "Barro Colorado Island photogrametry products: from 2018 to 2023 collected with the eBee drone."

## LandscapeScripts

### UAV_photogrametry.py
    The script UAV_photogrametry.py is an standard processing workflow in Agisoft Metashape 2.0 python API. The script takes a mission folder containing a subdirectory called Images. It process the images using key parameters such as higuest aligning setting, build point cloud in medium setting, filter points aggressively, build digital surface model and build orthomosaic. The exports are done in coordinate reference system UTM 17 N or epsg 32617, rasters are exported as GeoTIFF and point clouds as Polygon file format (.ply). This simplified code was used to process the whole Barro colorado island flights and the 50ha plot flights.

    Modules used:
        -pandas: 2.0.1                   
        -Metashape-2.0.3-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl

### 50ha_aligment.py
    The script 50ha_aligment.py is a complete workflow and performs multiple tasks such as combining, cropping, reprojecting, aligning locally, globally and performing elevation correction.
    It combines the orthomosaics and digital surface models into a single raster by reprojecting the DSM's to the shape of the orthomosaics and replace the alpha band with the DSM. Note: be sure to set the 4th band to none if the RGB doesnt looks right in your visualization software. 
    It crops all the combine products to buffered shape of the BCI 50ha plot. 20 meters were added to the xmax and ymax of the plot and 20 meters were substracted from the xmin and ymin of the plot. 
    The code performs the same task with the LIDAR orthomosaic by adding the lidar Digital elevation model and the digital terrain model to the 4th and 5th band. All the merging, reprojection and cropping is avaliable in the code and auxiliary files are provided in the data publication. 
    The orthomosaic from the P4P drone with theclosest date to the airborne LIDAR data collection over the 50ha plot is BCI_50ha_2023_05_23, this orthomosaic was locally aligned and the point with the higuest relaibility was used to globally align it to the Lidar orthomosaic by using the arosics module version 1.8.1, functions COREG and COREG_LOCAL.
    The closest date orthomosaic becomes the reference to the prior date. The prior date is aligned locally and then globally to then become the reference date to the next iteration of the for loop. The aligment is performed over the complete timeseries and it does not looses resolution. The global correction often fails for which we resort to the second point with the higuest reliability. Horizontal aligment is not perfect but it does mantain the data integrity without causing any warping and not resorting to resampling. 
    Finally, the same iterative approach is applied to the 4th band to correct the vertical misaligment of DSM's. We first resample the LIDAR DSM to match the drone DSM, the we calculate the delta of the median of both arrays and we either substract or add it to the drone DSM array to correct the difference in elevation and reduce the discrepancies. The corrected drone DSM becomes the reference and the prior date becomes the target for aligment until the for loop is completed.
    
    Note: local aligment is computationally intensive but it does generate nearly perfectly aligned orthomosaics of the 50ha plot. If you dont mind the warping and resampling that goes into this process please reach out to vasquezv@si.edu and I would be happy to share the locally aligned orthomosaics. Or you can use the code, it takes about 1 hr per orthomosaic in a XEON with 12 cores. 

    Modules used:
        -rasterio 1.2.10 py39h17c1fa0_0
        -numpy 1.25.1 pypi_0
        -geopandas 0.13.0 pypi_0
        -pandas 2.0.1 pypi_0
        -aroiscs 1.8.1
        -shapely 2.0.1 py39hd7f5953_0
        



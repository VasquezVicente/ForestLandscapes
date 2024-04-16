#one tree workflow
python
import os
import rasterio
import geopandas as gpd
import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.geometry import box, MultiPolygon
from rasterio.mask import mask
from shapely.ops import transform
from arosics import COREG_LOCAL, COREG
local_orthomosaics=r'D:\BCI_50ha\timeseries_local_alignment'
orthomosaics = [f for f in os.listdir(local_orthomosaics) if f.endswith('.tif')]
reference_orthomosaic = os.path.join(local_orthomosaics, orthomosaics[24])

reference_crownmap = r'D:\BCI_50ha\crown_segmentation\2020_08_01_improved.shp'
reference_crownmap = gpd.read_file(reference_crownmap) 

#select a random crown for testing
random_index= random.randint(0, len(reference_crownmap)-1)

#plot the crown and the orthomosaic with 5, 10, 20, 40 meter buffers
#objective to see which buffer size is best for the crown in order to local coreg
crown=reference_crownmap.iloc[random_index]
crown_geom = crown["geometry"]
geom_bounds = crown_geom.bounds
buffers = [5, 10, 20, 40]
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

for i, buffer in enumerate(buffers):
    ax = axs[i//2, i%2]
    geom_box = box(geom_bounds[0]-buffer, geom_bounds[1]-buffer, geom_bounds[2]+buffer, geom_bounds[3]+buffer)
    with rasterio.open(reference_orthomosaic) as src:
        out_image, out_transform = mask(src, [geom_box], crop=True)
        x_min, y_min = out_transform * (0, 0)
        xres, yres = out_transform[0], out_transform[4]
        if isinstance(crown_geom, MultiPolygon):
            pass
        transformed_geom = transform(lambda x, y: ((x-x_min)/xres, (y-y_min)/yres), crown_geom)
        ax.imshow(out_image.transpose((1, 2, 0))[:,:,0:3])
        ax.plot(*transformed_geom.exterior.xy, color='red')
        ax.axis('off')
plt.show()

#lets try with 20 meter buffer
#time series of the crown for a loop with the same overlayed geom. lets global coreg them all and local coreg them  all in test folder
for k in range(24, 0, -1):
    buffer=20
    geom_box = box(geom_bounds[0]-buffer, geom_bounds[1]-buffer, geom_bounds[2]+buffer, geom_bounds[3]+buffer)
    with rasterio.open(os.path.join(local_orthomosaics, orthomosaics[k])) as src:
        out_image, out_transform = mask(src, [geom_box], crop=True)
        x_min, y_min = out_transform * (0, 0)
        xres, yres = out_transform[0], out_transform[4]
        if isinstance(crown_geom, MultiPolygon):
            pass
        transformed_geom = transform(lambda x, y: ((x-x_min)/xres, (y-y_min)/yres), crown_geom)
        plt.imshow(out_image.transpose((1, 2, 0))[:,:,0:3])
        plt.plot(*transformed_geom.exterior.xy, color='red')
        plt.axis('off')
    plt.show()



#same type of loop to save the instances of the crown
for k in range(len(orthomosaics)):
    print(k)
    buffer=20
    geom_box = box(geom_bounds[0]-buffer, geom_bounds[1]-buffer, geom_bounds[2]+buffer, geom_bounds[3]+buffer)
    with rasterio.open(os.path.join(local_orthomosaics, orthomosaics[k])) as src:
        out_image, out_transform = mask(src, [geom_box], crop=True)
        x_min, y_min = out_transform * (0, 0)
        xres, yres = out_transform[0], out_transform[4]
        if isinstance(crown_geom, MultiPolygon):
            pass
        output_path = os.path.join(r"D:/BCI_50ha", 'test')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file=os.path.join(output_path, orthomosaics[k])
        with rasterio.open(output_file, 'w', driver='GTiff',
                           width=out_image.shape[2], height=out_image.shape[1], count=4,
                           dtype=out_image.dtype, crs=src.crs, transform=out_transform) as dst:
            dst.write(out_image)


#we list the files in the test folder
test_folder=r'D:/BCI_50ha/test'
test_files = [f for f in os.listdir(test_folder) if f.endswith('.tif')]

#backward alignment    
reference_crown=os.path.join(test_folder, test_files[24])
shift_calculations=[]
for date in test_files[23:23-10:-1]:
    print(date)
    target_crown=os.path.join(test_folder, date)

    output_image=os.path.join(r"D:/BCI_50ha/test/aligned_crowns",date)
    kwargs = {              'grid_res': 50,
                            'window_size': (512, 512),
                            'path_out': output_image,
                            'fmt_out': 'GTIFF',
                            'q': False,
                            'min_reliability':10,
                            'r_b4match': 4,
                            's_b4match': 4,
                            'max_shift': 200,
                            'nodata':(0, 0),
                            'match_gsd':False,
                        }
    CRL = COREG_LOCAL(reference_crown, target_crown, **kwargs)
    CRL.calculate_spatial_shifts()
    CRL.correct_shifts()
    table=CRL.CoRegPoints_table
    shifts = table[['ABS_SHIFT', 'RELIABILITY']]
    shifts = shifts.loc[(shifts['ABS_SHIFT'] != -9999) & (shifts['RELIABILITY'] >= 10)]
    shifts_score=((100-shifts['ABS_SHIFT'])*shifts['RELIABILITY'])/10000
    shift_score_mean=shifts_score.mean()
    #plt.hist(shifts_score, bins=20)
    #plt.show()
    shift_calculations.append(shift_score_mean)
    print(shift_score_mean)
    reference_crown=target_crown






#rubust normalization of orthomosaics rgb
#local alignment for indiviudual crwon failed with buffer 20 meter, orthomosaic 2019 month 12 does not hae good illumination
#lets try with buffer 40 meter
    
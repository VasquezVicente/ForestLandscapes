#create video of target Huras with line graph of predicted_leafing
import os
import pickle
import geopandas as gpd
import pandas as pd 

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_ortho=os.path.join(data_path,"orthomosaic_aligned_local")
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
crowns=gpd.read_file(path_crowns)
crowns['polygon_id']= crowns['GlobalID']+"_"+crowns['date'].str.replace("_","-")

#import the species we are trying to create videos of
sp= pd.read_csv(r'timeseries/dataset_predictions/dipteryx_sgbt.csv')

with open(r'timeseries/models/xgb_model.pkl', 'rb') as file:
      model = pickle.load(file)

X=sp[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]
Y= sp[['area', 'score', 'tag', 'GlobalID', 'iou',
       'date', 'latin', 'polygon_id']]

X_predicted=model.predict(X)
df_final = Y.copy()  # Copy Y to keep the same structure
df_final['leafing_predicted'] = X_predicted

dataset= df_final.merge(crowns[['polygon_id','geometry']],on='polygon_id',how='left')
dataset = gpd.GeoDataFrame(dataset, geometry='geometry', crs=crowns.crs)

#now we make the video for each of the globalIDS
from shapely.geometry import box
import rasterio
from rasterio.mask import mask
import shapely
import cv2
import tempfile
import matplotlib.pyplot as plt
import shutil

video_dir = os.path.join(data_path, "videos")
os.makedirs(video_dir, exist_ok=True)
all_globalids= dataset['GlobalID'].unique()

shutil.rmtree(temp_dir)
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Using temporary directory: {temp_dir}")

    frames = []

    for individual in all_globalids:
        indv_timeseries = dataset[dataset['GlobalID'] == individual]
        all_bounds = indv_timeseries.total_bounds
        absolute_box = box(all_bounds[0] - 5, all_bounds[1] - 5, all_bounds[2] + 5, all_bounds[3] + 5)

        for i, (_, row) in enumerate(indv_timeseries.iterrows()):
            print(f"Processing frame {i + 1} of {len(indv_timeseries)}")

            path_orthomosaic = os.path.join(data_path, 'orthomosaic_aligned_local', f"BCI_50ha_{row['date']}_local.tif")
            with rasterio.open(path_orthomosaic) as src:
                out_image, out_transform = mask(src, [absolute_box], crop=True)
                x_min, y_min = out_transform * (0, 0)
                xres, yres = out_transform[0], out_transform[4]

                transformed_geom = shapely.ops.transform(
                    lambda x, y: ((x - x_min) / xres, (y - y_min) / yres),
                    row.geometry
                )

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(out_image.transpose((1, 2, 0))[:, :, :3])
                ax.plot(*transformed_geom.exterior.xy, color='red')
                for interior in transformed_geom.interiors:
                    ax.plot(*interior.xy, color='red')
                ax.axis('off')

                # Save the frame
                frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
                fig.savefig(frame_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                frames.append(frame_path)

    # Create video from frames
        if frames:
            first_frame = cv2.imread(frames[0])
            height, width, _ = first_frame.shape
            video_path = os.path.join(video_dir, f"hura_{individual}.mp4")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width, height))  # 2 FPS

            for frame_file in frames:
                frame = cv2.imread(frame_file)
                out.write(frame)

            out.release()
            print(f"Saved video: {video_path}")

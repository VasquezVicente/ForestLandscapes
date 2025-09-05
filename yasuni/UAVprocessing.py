import os 
import geopandas as gpd
import pandas as pd
import shutil
import Metashape
from collections import defaultdict

project_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV"
dirs=[d for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path,d)) and d.startswith("ECUADOR_yasuni_")]

for idx, d in enumerate(dirs, 1):
    print(f"{idx}: {d}")

choice = int(input("Enter the number of the directory to process: ")) - 1
selected_dir = dirs[choice]
project_path = r"D:\Yasuni\ECUADOR_yasuni_2019_02_06_P4P\ECUADOR_yasuni_2019_02_06_P4P.psx"

doc = Metashape.Document()
doc.open(project_path)
chunk = doc.chunks[0]

out_path= os.path.join(os.path.dirname(project_path), "RGB")
os.makedirs(out_path, exist_ok=True)

used_names = defaultdict(int)
for photo in chunk.cameras:
    src = photo.photo.path
    base = os.path.basename(src)
    
    count = used_names[base]
    if count == 0:
        dst_name = base
    else:
        name, ext = os.path.splitext(base)
        dst_name = f"{name}_{count}{ext}"
    
    used_names[base] += 1
    dst = os.path.join(out_path, dst_name)
    shutil.copy(src, dst)

from skimage import io, exposure
ref_image = r"D:\Yasuni\ECUADOR_yasuni_2019_02_06_P4P\2019_Feb_06_Phantom_Flight_2_431.JPG"
ref_image= io.imread(ref_image)
#io.imread(os.path.join(out_path,"DJI_0053.JPG"))

# Folder of dark images
output_folder = out_path.replace('RGB', "normalized")
os.makedirs(output_folder, exist_ok=True)

jpg_files = [f for f in os.listdir(out_path) if f.lower().endswith('.jpg')]
total_files = len(jpg_files)

for i, filename in enumerate(jpg_files, 1):
    img_path = os.path.join(out_path, filename)
    img = io.imread(img_path)

    matched = exposure.match_histograms(img, ref_image, channel_axis=-1)

    out_file = os.path.join(output_folder, filename)
    io.imsave(out_file, matched.astype('uint8'), quality=100)

    print(f"[{i}/{total_files}] Processed: {filename}")

from shapely.geometry import box
path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\ECUADOR_yasuni_2025_05_30_M3E\20250530_yasuni_m3e_rgb_gr0p07_infer.shp"

crownmap= gpd.read_file(path)


minx, miny, maxx, maxy = crownmap.total_bounds

# Grid dimensions
cols = 6
rows = 3
tile_width = (maxx - minx) / cols
tile_height = (maxy - miny) / rows

# Create a list to store the tiles and their bounds
tiles = []
for i in range(cols):
    for j in range(rows):
        x0 = minx + i * tile_width
        y0 = miny + j * tile_height
        x1 = x0 + tile_width
        y1 = y0 + tile_height
        tile_geom = box(x0, y0, x1, y1)
        tiles.append({
            "tile_id": f"tile_{i}_{j}",
            "geometry": tile_geom
        })

# Convert tiles into a GeoDataFrame
tiles_gdf = gpd.GeoDataFrame(tiles, crs=crownmap.crs)

# For each tile, filter the crownmpa1 polygons that fall **completely** within the tile
output_dir = r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\ECUADOR_yasuni_2025_05_30_M3E\tiles2"
os.makedirs(output_dir, exist_ok=True)

for idx, tile in tiles_gdf.iterrows():
    tile_geom = tile.geometry
    tile_id = tile.tile_id

    # Select geometries that are fully within the tile
    in_tile = crownmap[crownmap.within(tile_geom)]

    if not in_tile.empty:
        out_fp = os.path.join(output_dir, f"{tile_id}.shp")
        in_tile.to_file(out_fp)

print("Done splitting into tiles.")

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import laspy

tile_size = 20  # Each tile is 20x20
buffer_size = 5  # 5m buffer
num_tiles_x = 5  # 5 tiles along x
num_tiles_y = 5  # 5 tiles along y

# Generate sub-bounds with a 5m buffer
sub_bounds = []

for i in range(num_tiles_x):
    for j in range(num_tiles_y):
        x_min = i * tile_size
        y_min = j * tile_size
        x_max = x_min + tile_size
        y_max = y_min + tile_size

        # Apply buffer
        x_min -= buffer_size
        y_min -= buffer_size
        x_max += buffer_size
        y_max += buffer_size

        sub_bounds.append((np.float64(x_min), np.float64(y_min), np.float64(x_max), np.float64(y_max)))


file=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\panama_BCI_plot1 0.010 m_clip.laz"
tile_size=20
with laspy.open(file) as file:
    writers = [None] * len(sub_bounds)  # List of writers for each sub-bound
    try:
        count = 0
        for points in file.chunk_iterator(10**6):
            print(f"{count / file.header.point_count * 100:.2f}%")

            # Ensure contiguous arrays for performance
            x, y = points.x.copy(), points.y.copy()

            point_piped = 0

            for i, (x_min, y_min, x_max, y_max) in enumerate(sub_bounds):
                mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

                if np.any(mask):
                    if writers[i] is None:
                        output_filename = f"retile_{int(x_min)}_{int(y_min)}.laz"
                        output_path = Path(r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\plot1") / output_filename
                        writers[i] = laspy.open(output_path, mode="w", header=file.header)

                    sub_points = points[mask]
                    writers[i].write_points(sub_points)

                point_piped += np.sum(mask)
                if point_piped == len(points):
                    break

            count += len(points)

        print(f"{count / file.header.point_count * 100:.2f}%")

    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
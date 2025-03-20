import laspy
import os

file=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\panama_BCI_plot2 0.010 m.las"
file_out=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\tiles"



with laspy.open(file) as f:
    for idx, points in enumerate(f.chunk_iterator(20000000), start=1):
        output_file = os.path.join(file_out, f"{idx}.laz")
        with laspy.open(output_file, mode="w", header=f.header) as writer:
            writer.write_points(points)
        print(f"Written chunk {idx} to {output_file}")



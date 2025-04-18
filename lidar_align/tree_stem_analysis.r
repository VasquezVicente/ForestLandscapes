library(lidR)
library(spanner)
#recall 1.chunkdatav2.py which chunks into 25 tiles with 5 meter buffer in all sides

tiles<-list.files("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1",pattern = "\\.laz$", full.names = TRUE)

# read the first tile corresponding to Q20 3020
Q3020<-readLAS(tiles[1])
# classify the ground points
Q3020<- classify_ground(Q3020,csf(cloth_resolution=0.3, rigidness=3L))
Q3020 = normalize_height(Q3020, tin())
Q3020 = classify_noise(Q3020, ivf(0.25, 3))
Q3020 = filter_poi(Q3020, Classification != LASNOISE)

myTreeLocs = get_raster_eigen_treelocs(las = Q3020, res = 0.05,
                                        pt_spacing = 0.0254,
                                        dens_threshold = 0.2,
                                        neigh_sizes = c(0.333, 0.166, 0.5),
                                        eigen_threshold = 0.5,
                                        grid_slice_min = 0.6666,
                                        grid_slice_max = 2.0,
                                        minimum_polygon_area = 0.025,
                                        cylinder_fit_type = "ransac",
                                        max_dia = 0.5,
                                        SDvert = 0.25,
                                        n_pts = 20,
                                        n_best = 25)

myTreeGraph = segment_graph(las = Q3020, tree.locations = myTreeLocs, k = 50,
                              distance.threshold = 0.5,
                              use.metabolic.scale = FALSE,
                              ptcloud_slice_min = 0.6666,
                              ptcloud_slice_max = 2.0,
                              subsample.graph = 0.1,
                              return.dense = FALSE)

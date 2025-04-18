library(pacman)
library(lidR)
library(sf)
library(terra)

#define the 3 TLS plots
#PLOT 2 coordinates at 870,270, transform shp to be at local coordinates for space reasons
plots<-read.csv('lidar_align/plot_data.csv')
shp_file <- "//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/shp/bci_20x20.shp"
shp_data <- st_read(shp_file)
plot(st_geometry(shp_data), main = "BCNM Lidar Raw Data - 20x20 Plots")

plots$xutm<-625773.86 + sqrt(plots$xcenter^2 + plots$ycenter^2) * cos (atan(plots$ycenter/plots$xcenter)-0.03246)
plots$yutm<- 1011775.84 + sqrt(plots$xcenter^2 + plots$ycenter^2) * sin (atan(plots$ycenter/plots$xcenter)-0.03246)

create_plot_polygon <- function(x_center, y_center, size = 100, angle = -0.03246) {
  half_size <- size / 2
  corners <- matrix(c(
    -half_size, -half_size,
    half_size, -half_size,
    half_size,  half_size,
    -half_size,  half_size,
    -half_size, -half_size
  ), ncol = 2, byrow = TRUE)
  rotation_matrix <- matrix(c(
    cos(angle), -sin(angle),
    sin(angle),  cos(angle)
  ), ncol = 2, byrow = TRUE)
  rotated_corners <- t(rotation_matrix %*% t(corners))
  rotated_corners[,1] <- rotated_corners[,1] + x_center
  rotated_corners[,2] <- rotated_corners[,2] + y_center
  st_polygon(list(rotated_corners))
}
plot_polygons <- st_sfc(lapply(1:nrow(plots), function(i) {
  create_plot_polygon(plots$xutm[i], plots$yutm[i])
}), crs = 32617)

plots_sf <- st_sf(plots, geometry = plot_polygons)
filtered_shp<-shp_data[st_intersects(shp_data, plots_sf$geometry[1], sparse = FALSE), ] 
plot(filtered_shp, col = NA, border = "red",lwd=5, main = "Plot Polygons", add=TRUE)
plot(plots_sf$geometry[1], col = NA, border = "blue",lwd=5, main = "Plot Polygons", add=TRUE)

filtered_shp<- filtered_shp[filtered_shp$X_IDX>29,]
filtered_shp<- filtered_shp[filtered_shp$Y_IDX<25,]

filtered_shp_transformed <- st_geometry(filtered_shp) - c(min(st_coordinates(filtered_shp)[,1]), 
                                                           min(st_coordinates(filtered_shp)[,2]))
theta <- 0.03246
R <- matrix(c(cos(theta), -sin(theta), sin(theta), cos(theta)), 2, 2)
rotated_geometry <- filtered_shp_transformed * R
min_x <- min(st_coordinates(rotated_geometry)[,1])
min_y <- min(st_coordinates(rotated_geometry)[,2])
final_geometry <- rotated_geometry - c(min_x, min_y)
plot(st_geometry(filtered_shp))
plot(plots_sf[1,])

###PLOT 1
ctg1<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1")
opt_chunk_size(ctg1)<- 20
opt_chunk_buffer(ctg1)<-5
plot(ctg1,chunk=TRUE)
plot(final_geometry, add=TRUE)


# classify ground points
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/classified/"
dir.create(outdir)
opt_output_files(ctg1) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
classified_ctg1 <- classify_ground(ctg1, csf(cloth_resolution=0.3, rigidness=3L))


classfied_ctg1<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/classified")
plot(classfied_ctg1,chunk=TRUE)
las_check(classfied_ctg1)
dtm_tls <- rast("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/classified/rasterize_terrain.vrt")
plot(dtm_tls, main="TLS Digital Terrain Model: Plot 1")

out_tiles="//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/decimated"
dir.create(out_tiles)
opt_output_files(classfied_ctg1) <- paste0(out_tiles, "/retile_{XLEFT}_{YBOTTOM}")
opt_chunk_buffer(classfied_ctg1)<-0
plot(classfied_ctg1, chunk=TRUE)
newctg<-decimate_points(classfied_ctg1, random(25000))

plot(newctg)
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/normalized/"
dir.create(outdir)
opt_chunk_buffer(newctg)<-10
opt_output_files(newctg) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
normalized_tls <- normalize_height(newctg,dtm)


normalized_tls<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/tLS/plot1/normalized")

opt_filter(normalized_tls)<-"-drop_z_above 40"
chm_tls<- rasterize_canopy(normalized_tls, 0.2, p2r(subcircle=0.20))
plot(chm_tls, main="TLS Canopy Height Model: Plot 1")



### ALS to check if we are in the correct plot
ctg1<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/ALS/roi")
dtm1<- rasterize_terrain(ctg1, 0.2, tin(), pkg="terra")
plot(dtm1,main="ALS Digital Terrain Model: Plot 1")
opt_output_files(ctg1)<- paste0("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/ALS/normalized/","/retile_{XLEFT}_{YBOTTOM}")
plot(ctg1)
normalized_tls <- normalize_height(ctg1,dtm1)
chm_als<- rasterize_canopy(normalized_tls,0.20,p2r(subcircle=0.2))
plot(chm_als,main="ALS Canopy Height Model: Plot 1")

plots_sf_buffered <- st_buffer(plots_sf[1,], dist = 5)
roi<- clip_roi(ctg1, plots_sf_buffered)
plot1_als<- readLAS(roi@data$filename[1])

normalized <- ctg1 - dtm1
plot(normalized)



###MLS work catalog

mls_ctg<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/MLS/plot1")
plot(mls_ctg)
out_tiles="//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/MLS/plot1/decimated"
dir.create(out_tiles)
opt_output_files(mls_ctg) <- paste0(out_tiles, "/retile_{XLEFT}_{YBOTTOM}")
opt_chunk_buffer(mls_ctg)<-5
opt_chunk_size(mls_ctg)<-20

plot(mls_ctg, chunk=TRUE)
newctg<-decimate_points(mls_ctg, homogenize(10000,1))


outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/MLS/plot1/classified/"
dir.create(outdir)
opt_output_files(newctg) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
classified_ctg1 <- classify_ground(newctg, csf(cloth_resolution=0.3, rigidness=3L))

dtm<- rasterize_terrain(classified_ctg1,res=0.20,tin())
plot(dtm, main="MLS Digital Terrain Model: plot 1",size=10)


classified_ctg1<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/MLS/plot1/classified")
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/MLS/plot1/normalized/"
dir.create(outdir)
opt_chunk_buffer(classified_ctg1)<-10
opt_output_files(classified_ctg1) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
normalized_mls <- normalize_height(classified_ctg1,dtm)

chm_mls<- rasterize_canopy(normalized_mls, 0.20, p2r(subcircle=0.20))
plot(chm_mls,main="MLS Canopy Height Model: plot 1")


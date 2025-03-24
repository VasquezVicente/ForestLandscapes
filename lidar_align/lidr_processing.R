library(pacman)
# install.packages("devtools")
devtools::install_github("lmterryn/ITSMe", build_vignettes = TRUE)
p_load(lidR, mapview,sp, sf, RCSF, reticulate,remotes,lidUrb,ForestClassR,TreeLS,ITSMe)

use_virtualenv("lidUrb", required = TRUE)
setup_env()

remotes::install_github('tiagodc/TreeLS')
library(TreeLS)

ctg <- readLAScatalog("D:/TLS/tiles")
las_check(ctg)
plot(ctg)
ctg

#First get rid of the noise
out_tiles="D:/TLS/tiles_decimated"
dir.create(out_tiles)
opt_output_files(ctg) <- paste0(out_tiles, "/retile_{XLEFT}_{YBOTTOM}")
newctg<-decimate_points(ctg, random(30000))


#classify the noise
out_tiles="D:/TLS/tiles_noise"
dir.create(out_tiles)
opt_output_files(newctg)<- paste0(out_tiles,"/retile_{XLEFT}_{YBOTTOM}")
opt_chunk_buffer(newctg) <- 10
newctg<-classify_noise(newctg, algorithm=sor(k = 10, m = 3, quantile = FALSE))


#classify the ground points
out_tiles="D:/TLS/tiles_ground"
dir.create(out_tiles)
opt_output_files(newctg) <- paste0(out_tiles, "/retile_{XLEFT}_{YBOTTOM}")
opt_chunk_buffer(newctg)<- 10
ground<-classify_ground(newctg,csf(cloth_resolution=0.5, rigidness=2L))

#remove the noise
no_noise<-readLAScatalog("D:/TLS/tiles_ground")
opt_filter(no_noise) <- "-drop_class 18"
plot(no_noise)


#define the ROI
#PLOT 2 coordinates at 870,270, transform shp to be at local coordinates for space reasons
x=650
y=450
x_utm = 625773.86 + sqrt(x^2 + y^2) * cos (atan(y/x)-0.03246)
y_utm = 1011775.84 + sqrt(x^2 + y^2) * sin (atan(y/x)-0.03246)

shp_file <- "//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/shp/bci_20x20.shp"
shp_data <- st_read(shp_file)
shp_data_transformed <- shp_data
shp_data_transformed$geometry <- st_geometry(shp_data) - c(x_utm - 50, y_utm - 50)

plot(st_geometry(shp_data_transformed), border = "blue", col = NA, main = "Transformed Shapefile", add=TRUE)
points(50, 50, col = "red", pch = 19, cex = 2)  

##100X100 PLOT WITH 5METER BUFFER
e <- st_as_sf(as(raster::extent(0, 100, 0, 100), "SpatialPolygons"))
st_crs(e) <- st_crs(shp_data_transformed)
filtered_shp <- shp_data_transformed[st_intersects(shp_data_transformed, e, sparse = FALSE), ]
filtered_shp <- filtered_shp[filtered_shp$X_IDX!=29,]
filtered_shp <- filtered_shp[filtered_shp$X_IDX!=35,]
filtered_shp <- filtered_shp[filtered_shp$Y_IDX!=19,]
filtered_shp <- filtered_shp[filtered_shp$Y_IDX!=25,]


# Plot the result
plot(st_geometry(filtered_shp), border = "blue", col = NA, main = "Filtered Polygons", add=TRUE, label=TRUE)
plot(e, add = TRUE, border= "red") # Add the extent box in red



path="D:/TLS/20x20_normalize"
ctg<- readLAScatalog(path) #read the normalized point cloud data
plot(ctg)
plot(ctg$geometry[1],border="red",add=TRUE)
ctg$filename[1]
$30-23


als<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/ALS")
plot(als)
tls_20x20_plots<-filtered_shp$P20
als_filtered_plots<- shp_data[shp_data$P20 %in% tls_20x20_plots,]
plot(als_filtered_plots, add=TRUE)
out_tiles<-"D:/ALS"
dir.create(out_tiles)
opt_output_files(als) <- paste0(out_tiles, "/retile_{XLEFT}_{YBOTTOM}")
opt_chunk_buffer(als)<- 10
als_20x20<- clip_roi(als,als_filtered_plots)
plot(als_20x20)

plot(als_20x20$geometry[1],border="red",add=TRUE)
als_20x20$filename[1]


tile<-readLAS(ctg$filename[2]) #read tile #1
segmented  <- LW_segmentation_graph(tile)
ground<- filter_poi(segmented, Classification==2) # divided the point cloud into ground cloud and segmented
segmented<-filter_poi(segmented, Classification!=2)

wood_leaf_palette <- c("brown","green")
segmented@data[, label := as.numeric(p_wood >= 0.96)]
lidR::plot(segmented, color="label", size=2, pal = wood_leaf_palette)

wood <- filter_poi(segmented, Classification != label)
plot(wood)

map= treeMap(wood, map.eigen.knn(),0)
x=plot(wood, color="label", size=0.1, pal = wood_leaf_palette)
add_treeMap(x, map, color='yellow', size=2)

plot(map)



tls = treePoints(wood, map, trp.crop())

add_treePoints(x, tls, size=1)
add_treeIDs(x, tls, cex = 2, col='yellow')

all_trees <- unique(tls@data$TreeID)
results <- list()  # List to store results
plot(tls[tls$TreeID==3,])


for (tree in all_trees) {
  tryCatch({
    # Extract individual tree data
    indv <- tls[tls$TreeID == tree, ]
    plot(indv)
    
    # Apply noise classification and filtering
    indv <- classify_noise(indv, algorithm = sor(k = 10, m = 3, quantile = FALSE))
    indv <- filter_poi(indv, Classification != 18)
    
    # Convert to data frame
    indv_df <- data.frame(X = indv@data$X, Y = indv@data$Y, Z = indv@data$Z)
    
    # Compute DBH and center coordinates
    out_dbh <- dbh_pc(pc = indv_df, plot = TRUE)
    centerx <- out_dbh$plot$layers[[2]]$data$x0
    centerY <- out_dbh$plot$layers[[2]]$data$y0
    x_center_utm <- x_utm + (centerx - 50)
    y_center_utm <- y_utm + (centerY - 50)
    
    # Store results in a list
    results[[length(results) + 1]] <- data.frame(
      ID = tree,
      X = x_center_utm,
      Y = y_center_utm,
      DBH = out_dbh$dbh,
      R2 = out_dbh$R2
    )
  }, error = function(e) {
    message(paste("Error processing Tree ID:", tree, "-", e$message))
  })
}

# Combine all results into a single data frame
if (length(results) > 0) {
  tree_df <- do.call(rbind, results)
  
  # Convert to sf object
  point_sf <- st_as_sf(tree_df, coords = c("X", "Y"), crs = 32617)
  
  # Export the shapefile
  st_write(point_sf, "D:/TLS/tree_center.shp", delete_layer = TRUE)
  
  print("Shapefile exported successfully!")
} else {
  print("No valid tree data processed.")
}


#BCI tree stems
bci_stems<-bci.tree8
bci_stems$utmx<-625773.86 + sqrt(bci_stems$gx^2 + bci_stems$gy^2) * cos (atan(bci_stems$gy/bci_stems$gx)-0.03246)
bci_stems$utmy<-1011775.84 + sqrt(bci_stems$gx^2 + bci_stems$gy^2) * sin (atan(bci_stems$gy/bci_stems$gx)-0.03246)
bci_stems<-bci_stems[!is.na(bci_stems$utmx),]
point_sf <- st_as_sf(bci_stems, coords = c("utmx", "utmy"), crs = 32617)
st_write(point_sf, "D:/TLS/bci_stems.shp")
# Access the dbh, residual and fdbh values from the output list
dbh <- out_dbh$dbh
residual_dbh <- out_dbh$R2
fdbh <- out_dbh$fdbh
# Use dab_pc function with default parameters and plot the fit
out_dab <- dab_pc(pc = pc_tree, plot = TRUE)
# Access the dab, residual and fdab values from the output list
ddab <- out_dab$dab
residual_dab <- out_dab$R2
fdab <- out_dab$fdab



library(lidUrb)      
library(ForestClassR) 
install_github("lucasbielak/ForestClassR")

virtualenv_create("lidUrb", packages="jakteristics", pip=TRUE)

use_virtualenv("lidUrb", required = TRUE)
plot(tile)

segmented  <- LW_segmentation_graph(tile)
lidR::plot(segmented, color = "p_wood", legend = TRUE)

# Classification based on thresholds
wood_leaf_palette <- c("chartreuse4", "cornsilk2") # palette Dark green for leaves, light for wood

# DBSCAN-based wood classification using p_wood threshold
segmented@data[, label := as.numeric(p_wood >= 0.96)]
lidR::plot(segmented, color="label", size=2, pal = wood_leaf_palette)

#plot only wood

wood <- filter_poi(segmented, Classification != label)
plot(wood)


wood_no_ground<- filter_poi(wood, Classification != 2)
plot(wood_no_ground)


las_eigen <- features_jak(segmented, radius = 1)





dtm <- rasterize_terrain(no_noise, 0.2, tin(), pkg = "terra")







out_tiles="D:/TLS/tiles_no_noise"
dir.create(out_tiles)
opt_output_files(ground) <- paste0(out_tiles, "/retile_{XLEFT}_{YBOTTOM}")
opt_chunk_buffer(ground)<- 10
no_noise=filter_duplicates(ground, Classification != 18)




plot(ctg, chunk = TRUE)
opt_output_files(ctg) <- paste0(out_tiles, "/retile_{XLEFT}_{YBOTTOM}")
newctg <- filter_duplicates(ctg)

catalog_retile(ctg)




## WE ARE GOING TO FILTER THE NOISE
plot2919= clip_roi(ctg,filtered_shp$geometry[1])
filtered_shp$geometry[1]
filtered_shp$Label[1]

plot(plot2919)


filtered <- filter_duplicates(plot2919)

rm(plot2919)
rm(filtered)
las <- classify_noise(filtered, algorithm=sor(k = 10, m = 3, quantile = FALSE))
las <- classify_ground(las, csf(cloth_resolution=2, rigidness=3L))
las <- filter_poi(las, Classification != 18)
thinned1 <- decimate_points(las, random(10000))
ground <- filter_poi(las, Classification == 2)

plot(ground)
plot(thinned1)
#plot(las)

#Classifying noise points by Statistical Outliers Removal (SOR)


#Filtering out the noise points (Class 18)
las <- filter_poi(las, Classification != 18)


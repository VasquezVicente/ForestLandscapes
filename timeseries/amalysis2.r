
library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan,circular, bpnreg,tidyr, tidyverse, bpnreg, sf,dbscan)

# Convert to DOY
radians_to_days <- function(r) {
  (r / (2 * pi)) * 365
}

# Load the data
ceiba <- read.csv("timeseries/dataset_analysis/ceiba_analysis.csv")
ceiba$date<- as.Date(ceiba$date, format= "%Y-%m-%d")
ceiba$latin<- "Ceiba pentandra"

hura <- read.csv("timeseries/dataset_analysis/hura_analysis.csv")
hura$date<-as.Date(hura$date, format= "%Y-%m-%d")
hura$latin<- "Hura crepitans"

dipteryx <- read.csv("timeseries/dataset_analysis/dipteryx_analysis.csv")
dipteryx$date<-as.Date(dipteryx$date, format= "%Y-%m-%d")
dipteryx$latin<- "Dipteryx oleifera"

jacaranda <- read.csv("timeseries/dataset_analysis/jacaranda_analysis.csv")
jacaranda$date<-as.Date(jacaranda$date, format= "%Y-%m-%d")
jacaranda$latin<- "Jacaranda copaia"

cavallinesia <- read.csv("timeseries/dataset_analysis/cavallinesia_analysis.csv")
cavallinesia$date<-as.Date(cavallinesia$date, format= "%Y-%m-%d")
cavallinesia$latin<- "Cavallinesia platanifolia"

cavallinesia <- read.csv("timeseries/dataset_analysis/cavallinesia_analysis.csv")
cavallinesia$date<-as.Date(cavallinesia$date, format= "%Y-%m-%d")
cavallinesia$latin<- "Cavallinesia platanifolia"

quararibea <- read.csv("timeseries/dataset_analysis/quararibea_analysis.csv")
quararibea$date<-as.Date(quararibea$date, format= "%Y-%m-%d")
quararibea$latin<- "Quararibea stenophylla"
quararibea$isFlowering_predicted<- as.character(quararibea$isFlowering_predicted)

all<- bind_rows(ceiba,hura,dipteryx,jacaranda, cavallinesia, quararibea)
all$dayYear <- yday(all$date)


########## DBSCAN and removal of repeated breakpoints############
##focusing on Hura crepitans and start of leaf drop

sp="Dipteryx oleifera"
cava_bad<-c("87757645-6a6c-41c0-b862-6b192c82f4cb","6e7eb081-2a27-44a7-85f2-a88a40e3e53a","08c4ec87-3de9-40d7-b64e-d3a85534739e")
hura_bad<-c("afbda3c3-9e08-40c8-a682-d6c87c4dd38a","cd5e49bb-fab1-41b0-ab35-74fbdcd2d270","f8ab748b-e750-4f6f-8ec7-cd1c06ae7a08",
           "6a82b639-5ded-4f78-8223-ed4bf4f53e27","fe79536f-a0ab-415b-8660-807cb81acfa5","e956f57c-cd96-41a4-ad4b-d1b458abf281",
           "d1aa2f45-83cf-4f35-8f66-f531e2082ce9","cd5e49bb-fab1-41b0-ab35-74fbdcd2d270","afbda3c3-9e08-40c8-a682-d6c87c4dd38a",
           "a73ea74d-0b13-40d1-9f84-6eb44c0afbb3","79662f81-ff7d-4dad-9545-383646f4c747","3542b534-9007-4e11-890d-fd260413d31b",
           "854b05aa-129f-4af3-948f-bce96af487ae","553c3929-d889-4c5f-947e-7c72d18d1b76","153c0ffb-3aa9-4ccc-a4b2-e95f819d7e40",
           "41a365e1-2af7-4963-b811-0b4f748bbfa6","25cb7d21-26a0-44ec-894a-ddf163db09ee","18cb1282-1df7-42fb-968a-5bbd8b009c8a",
           "6a82b639-5ded-4f78-8223-ed4bf4f53e27","2b4849fd-67b7-4f1b-a683-886db505b5a9","1bb104f9-9cfd-4761-bd2c-0e326ac68f0e",
           "abf712a6-4086-4090-8433-0ea8986fc980"
           )

alseis_bad<- c("a9fbce15-5d86-432a-93a4-9ee6f0bfb19e","ad86904d-84dd-4a1b-9030-d63c1ca03ef4")

quara_bad<- c()

hura_filter<-all %>% filter(latin==sp & !GlobalID %in% hura_bad )

tag_lookup <- hura_filter %>%
  filter(!is.na(tag)) %>%
  distinct(GlobalID, tag)

hura_fixed <- hura_filter %>%
  select(-tag) %>%  # Remove old (possibly NA) tag column
  left_join(tag_lookup, by = "GlobalID")  

#check the liana load
crownmap<- "D:/BCI_50ha_2022_2023_crownmap.shp"
crownmap_sf <- st_read("D:/BCI_50ha_2022_2023_crownmap.shp", quiet = TRUE)
crownmap_df <- st_drop_geometry(crownmap_sf)

hura_fixed$tag<- as.character(hura_filter$tag)
crownmap_df$tag<-as.character(crownmap_df$tag)

merged <- hura_fixed %>% 
  left_join(crownmap_df, by = "tag")

View(merged)

liana_lookup <- merged %>%
  filter(!is.na(lianas)) %>%
  distinct(GlobalID, lianas)

# Step 2: Join known lianas to merged
merged_fixed <- merged %>%
  left_join(liana_lookup, by = "GlobalID", suffix = c("", ".lookup")) %>%
  mutate(
    lianas = if_else(is.na(lianas), lianas.lookup, lianas)
  ) %>%
  select(-lianas.lookup)  # clean up temporary column
View(merged_fixed)
windows()
ggplot(merged_fixed, aes(x = GlobalID, y = leafing, color = lianas)) +
  geom_boxplot()+
  theme(axis.text.x=element_blank())

View(hura_mean)
## there is trees that remain with leaves
evergreens <- merged %>%
  group_by(GlobalID) %>%
  summarise(min_leafing = min(leafing, na.rm = TRUE)) %>%
  filter(min_leafing >= 85) %>%
  pull(GlobalID)
#print the evergreen trees
print(evergreens)

#check the liana load
crownmap<- "D:/BCI_50ha_2022_2023_crownmap.shp"
crownmap_sf <- st_read("D:/BCI_50ha_2022_2023_crownmap.shp", quiet = TRUE)
crownmap_df <- st_drop_geometry(crownmap_sf)

View(crownmap_df)




###the real stuff
hura_clustered <- all %>%
  filter(break_type == "start_leaf_drop")%>%
  filter(latin==sp) %>% filter(!GlobalID %in% cava_bad) %>% filter(!GlobalID %in% evergreens)

remove.packages("dbscan")
install.packages('dbscan')

#cluster the leaf drop breakpoints
db <- dbscan::dbscan(as.matrix(hura_clustered$date_num), eps = 40, minPts = 10)  ##140 for cavallinesia, 40 for hura  #25 for quararibea
hura_clustered$leaf_drop_cluster <- db$cluster

# check the clustering
windows()
hura_clustered%>%ggplot(aes(x = date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)

#More than 2 breakpoint per phenological year should be average between them
hura_clustered$leafing<- as.numeric(hura_clustered$leafing)
base_date <- as.Date("2018-04-04")
hura_mean <- hura_clustered %>% 
  group_by(GlobalID, leaf_drop_cluster) %>% 
  summarise(
    date_num_mean = mean(date_num, na.rm = TRUE),
    mean_leafing= mean(leafing),
    .groups = "drop"
  ) %>% 
  mutate(
    mean_date  = as.Date(base_date + round(date_num_mean)),
    year = as.integer(format(mean_date, "%Y")),
    is_leap = (year %% 4 == 0 & year %% 100 != 0) | (year %% 400 == 0),
    dayOfYear  = yday(mean_date),
    days_in_year = ifelse(is_leap, 366, 365),
    theta_rad = dayOfYear * 2 * pi / days_in_year,
    mean_leafing=mean_leafing
  )

hura_mean<- hura_mean %>% filter(leaf_drop_cluster !=0)

hura_mean$leaf_drop_cluster <- recode(hura_mean$leaf_drop_cluster,
                                      `1` = "2018-2019",
                                      `2` = "2019-2020",
                                      `3` = "2020-2021",
                                      `4` = "2021-2022",
                                      `5` = "2022-2023",
                                      `6` = "2023-2024")

hura_mean$leaf_drop_cluster <- factor(
  hura_mean$leaf_drop_cluster,
  levels = c("2018-2019", "2019-2020", "2020-2021", 
             "2021-2022", "2022-2023", "2023-2024")
)

windows()
hura_mean%>%ggplot(aes(x = mean_date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)

windows()
hist(hura_mean$mean_leafing, breaks=20)



###########################################
options("contrasts")
options(contrasts = c("contr.treatment", "contr.poly")) # set contrasts to treatment contrasts for categorical variables

# good lets fit the damn leaf drop model
hura_mean$leaf_drop_cluster <- as.factor(hura_mean$leaf_drop_cluster) #important
hura_mean$GlobalID<- as.factor(hura_mean$GlobalID)
hura_mean$GlobalID_num <- as.numeric(as.factor(hura_mean$GlobalID))
hura_mean$leaf_drop_cluster_num <- as.numeric(as.factor(hura_mean$leaf_drop_cluster))

##########GRAND MODEL##########
grand_model<- bpnr(pred.I = theta_rad ~ 1,
                   data = hura_mean,
                   its = 2000,
                   burn = 200,
                   n.lag = 5
)
lin_grand_model<- as_tibble(coef_lin(grand_model), rownames = "term")
radians_to_days(atan2(pull(lin_grand_model, mean)[2], pull(lin_grand_model, mean)[1]) %% (2 * pi))

###### MODEL WITHIN SPECIES VARIATION IN LEAF DROP TIMING ##########

grand_indv_model <- bpnr(
  pred.I = theta_rad ~ GlobalID,
  data = hura_mean,
  its = 10000,
  burn = 700,
  n.lag = 5
)

grand_indv_model_tbl <- as_tibble(grand_indv_model$circ.coef.means, rownames = "term")
grand_indv_model_tbl$DOY <- radians_to_days(grand_indv_model_tbl$mean %% (2 * pi))
grand_indv_model_tbl<- grand_indv_model_tbl[1:28,] #13 individuals for cava
grand_indv_model_tbl$mean<- circular(grand_indv_model_tbl$mean, units="radians", modulo="2pi")
View(grand_indv_model_tbl)

M <- as.matrix(grand_indv_model$beta1)
indv <- colnames(M)[1:28]

B01 <- grand_indv_model$beta1[, indv[1]]
B02 <- grand_indv_model$beta2[, indv[1]]
results_indv <- data.frame(
  row.names = indv,
  mode_angle = numeric(length(indv)),
  mean_angle = numeric(length(indv)),
  sd_angle = numeric(length(indv)),
  LB_angle = numeric(length(indv)),
  UB_angle = numeric(length(indv)),
  mode_var = numeric(length(indv)),
  mean_var = numeric(length(indv)),
  sd_var = numeric(length(indv)),
  LB_var = numeric(length(indv)),
  UB_var = numeric(length(indv))  
)

for (i in seq_along(indv)) {
  inv <- indv[i]
  b1 <- grand_indv_model$beta1[, inv]
  b2 <- grand_indv_model$beta2[, inv]
  aM <- atan2(B02 + b2, B01 + b1)
  zeta_probe <- sqrt((B01 + b1)^2 + (B02 + b2)^2)^2 / 4
  var_probe <- 1 - sqrt((pi * zeta_probe) / 2) * exp(-zeta_probe) *
    (besselI(zeta_probe, 0) + besselI(zeta_probe, 1))
  results_indv[inv, ] <- c(
    mode_est(circular(aM,modulo="2pi")),
    mean(circular(aM,modulo="2pi")),
    sd(aM),
    hpd_est(aM),
    mode_est(var_probe),
    mean(var_probe),
    sd(var_probe),
    hpd_est(var_probe)
  )
}

results_indv$DOY<- radians_to_days(results_indv$mean_angle%% (2*pi))
results_indv <- results_indv %>%
  tibble::rownames_to_column(var = "Individual")
View(results_indv)
levels(hura_mean$GlobalID)
results_indv_clean <- results_indv %>%
  mutate(
    Individual = str_remove(Individual, "^GlobalID"),  # Remove "GlobalID" at start
    Individual = ifelse(Individual == "(Intercept)", "078e0ab3-42ac-4a64-b96b-09cdf590f024", Individual)
  )

View(results_indv_clean)

###### MODEL WITHIN LEAF DROP CLUSTER AS FIXED EFFECT#########
grand_leaf_model<- bpnr(
  pred.I= theta_rad ~leaf_drop_cluster,
  data = hura_mean,
  its = 2000,
  burn = 700,
  n.lag = 5
)

grand_leaf_model_tbl<- as_tibble(grand_leaf_model$circ.coef.means, rownames= "term")
grand_leaf_model_tbl$DOY<- radians_to_days(grand_leaf_model_tbl$mean %% (2*pi))
grand_leaf_model_tbl$mean<- circular(grand_leaf_model_tbl$mean, units='radians', modulo="2pi")
grand_leaf_model_tbl<- grand_leaf_model_tbl[1:6,]
View(grand_leaf_model_tbl)

###plot for the exploratory models#############################
grand_mean <- data.frame(mean = atan2(pull(lin_grand_model, mean)[2], pull(lin_grand_model, mean)[1])%% (2*pi))
grand_mean$mean<- circular(grand_mean$mean, units = "radians", modulo = "2pi")

plot_data <- left_join(
  hura_mean, 
  results_indv_clean %>% select(Individual, mean_var),
  by = c("GlobalID" = "Individual")
)
View(plot_data)
plot_data$point_height <- 1 - plot_data$mean_var


windows()
ggplot() +
  geom_point(data = plot_data, aes(x = theta_rad, y = point_height, shape=leaf_drop_cluster), 
             size = 2, alpha = 0.6) +
  coord_polar(start = 0, direction = 1) + 
  geom_segment(data = grand_mean,  
               aes(x = mean, y = 0, xend = mean, yend = 1.2), 
               arrow = arrow(length = unit(0.2, "cm")), 
               size = 1) +
  geom_segment(data = results_indv,
               aes(x = mean_angle, y = 0, xend = mean_angle, yend = 1 - mean_var, color = Individual), 
               arrow = arrow(length = unit(0.2, "cm")), 
               size = 1, show.legend=FALSE)+
  geom_segment(
    data = results,
    aes(x = mean_angle, y = 0, xend = mean_angle, yend = (1-mean_var)+0.2), 
    size = 1 ) +
  geom_point(data = results,
  aes(x = mean_angle, y = (1-mean_var)+0.2,shape = pheno_year), 
  size = 3, stroke = 1.2)+
  scale_y_continuous(breaks=NULL)+
  scale_x_continuous(limits = c(0, 2 * pi),
    breaks = 2 * pi * (cumsum(c(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30)) + 15) / 365,
    labels = month.abb
  ) +
  labs(x = NULL, y = NULL,
    title = paste("Mean Angle and Mean Resultant Length (", sp, ")", sep = ""),
    subtitle = paste(
    "Shapes = observations per year\n",
    "Colored arrows = individual mean angles and concentrations\n",
    "Shaped arrows = yearly mean angles and concentrations"
  )
  )+
  theme_minimal()+
  theme(legend.position="bottom",
  axis.text.x = element_text(size = 12,color='black', face='bold'), 
  panel.grid.major = element_line(linewidth=1),
  panel.grid.minor = element_line(linewidth = 1),
  plot.title = element_text(hjust = 0.5, size=19,face='bold'),
  plot.subtitle = element_text(hjust = 0.5,size=12),
  legend.text = element_text(size = 15),                   # increase legend text
  legend.title = element_text(size = 17),)+
  labs(shape="Year")

### within species variation in leaf drop timing
### adding the leaf drop cluster as a fixed effect
fit_leafing <- bpnme(pred.I = theta_rad ~ leaf_drop_cluster + (1 | GlobalID_num),
                    data = hura_mean,
                    its = 10000,
                    burn = 700,
                    n.lag = 5
)

windows()
traceplot(fit_leafing, parameter= "beta2")

## i need the mean differences
##pull the first 6 column names of beta1
M <- as.matrix(fit_leafing$beta1)
pheno_years <- colnames(M)[1:6]

# Intercepts
B01 <- fit_leafing$beta1[, pheno_years[1]]
B02 <- fit_leafing$beta2[, pheno_years[1]]

# Mode function
posterior_mode <- function(x) {
  d <- density(x)
  d$x[which.max(d$y)]
}

# Empty list to store results
results_list <- list()

# Loop over all unique year pairs
combs <- combn(pheno_years, 2, simplify = FALSE)  # skip intercept

for (i in seq_along(combs)) {
  year_a <- combs[[i]][1]
  year_b <- combs[[i]][2]
  
  # Get coefficients
  a1 <- fit_leafing$beta1[, year_a]
  a2 <- fit_leafing$beta2[, year_a]
  
  b1 <- fit_leafing$beta1[, year_b]
  b2 <- fit_leafing$beta2[, year_b]
  
  # Compute circular means
  aM <- atan2(B02 + a2, B01 + a1)
  bM <- atan2(B02 + b2, B01 + b1)

  # Posterior difference
  diff_post <- aM - bM

  # circular vairance
  zeta_probe <- sqrt((B01 + b1)^2 + (B02 + b2)^2)^2/4
  var_probe  <- 1 - sqrt((pi * zeta_probe)/2) * exp(-zeta_probe) *
                          (besselI(zeta_probe, 0) + besselI(zeta_probe, 1))
    
  # Plot histograms of aM and bM
  windows()
  hist(bM, breaks = 30, col = rgb(0, 0, 1, 0.5), xlim = c(-pi/3, pi/3),
       main = paste0("Hist: ", year_a, " (aM) vs ", year_b, " (bM)"),
       xlab = "Value")
  hist(aM, breaks = 30, col = rgb(1, 0, 0, 0.5), add = TRUE)
  legend("topright", legend = c(paste0("bM: ", year_b), paste0("aM: ", year_a)),
         fill = c(rgb(0, 0, 1, 0.5), rgb(1, 0, 0, 0.5)))
  
  # Plot histogram of diff_post
  windows()
  hist(diff_post, breaks = 50, col = "gray",
       main = paste0("Posterior of ", year_a, " - ", year_b),
       xlab = paste0("Difference (", year_a, " - ", year_b, ")"))
  abline(v = 0, col = "red", lty = 2)
  mean_diff <- mean(diff_post)
mode_diff <- posterior_mode(diff_post)
sd_diff   <- sd(diff_post)
lb_diff   <- quantile(diff_post, 0.025)
ub_diff   <- quantile(diff_post, 0.975)

# Probabilities relative to direction of effect
if (mean_diff > 0) {
  p_effect_direction <- mean(diff_post > 0)
} else {
  p_effect_direction <- mean(diff_post < 0)
}
p_opposite_direction <- 1 - p_effect_direction

# Save to list
results_list[[i]] <- data.frame(
  YearA = year_a,
  YearB = year_b,
  mean = mean_diff,
  mode = mode_diff,
  sd   = sd_diff,
  LB   = lb_diff,
  UB   = ub_diff,
  DAY  = radians_to_days(mean_diff),
  P_effect_direction = p_effect_direction,
  P_opposite_direction = p_opposite_direction
)
}

# Combine all into one data frame
summary_all_pairs <- do.call(rbind, results_list)

# Print final summary
summary_all_pairs$significance <- ifelse(
  (summary_all_pairs$LB > 0 & summary_all_pairs$UB > 0) |
  (summary_all_pairs$LB < 0 & summary_all_pairs$UB < 0),
  "*",
  ""
)
summary_all_pairs$label_text <- paste0(round(summary_all_pairs$DAY, 0), summary_all_pairs$significance)
View(summary_all_pairs)

windows()
ggplot(summary_all_pairs, aes(x = YearB, y = YearA, fill = DAY, label = label_text)) +
  geom_tile() +
  geom_text(size= 6) +
  scale_fill_gradient2(low = "red2", mid = "white", high = "blue", midpoint = 0,
    name = "Difference in days") +
  scale_x_discrete(
    name = "Comparison Year",
    labels = c("2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024")
  ) +
  scale_y_discrete(
    name = "Reference Year",
    labels = c("2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024")
  ) +
  theme_minimal()+
  labs(
    title = paste("Difference between all phenological year pairs (", sp, ")", sep = "")
    
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size=12),
    axis.text.y = element_text(angle = 45, hjust = 1, size=12),
    axis.title.x = element_text(size = 15, face = "bold", color = "black"),
    axis.title.y = element_text(size = 15, face = "bold", color = "black"),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20)
  )
##calculate the circular variance of all the groups 

results <- data.frame(
  row.names = pheno_years,
  mode_angle = numeric(length(pheno_years)),
  mean_angle = numeric(length(pheno_years)),
  sd_angle = numeric(length(pheno_years)),
  LB_angle = numeric(length(pheno_years)),
  UB_angle = numeric(length(pheno_years)),
  mode_var = numeric(length(pheno_years)),
  mean_var = numeric(length(pheno_years)),
  sd_var = numeric(length(pheno_years)),
  LB_var = numeric(length(pheno_years)),
  UB_var = numeric(length(pheno_years))  
)
B01 <- fit_leafing$beta1[, pheno_years[1]]
B02 <- fit_leafing$beta2[, pheno_years[1]]


for (i in seq_along(pheno_years)) {
  yr <- pheno_years[i]
  
  b1 <- fit_leafing$beta1[, yr]
  b2 <- fit_leafing$beta2[, yr]
  
  # Circular mean angle (posterior samples)
  aM <- atan2(B02 + b2, B01 + b1)
  
  # Circular variance
  zeta_probe <- sqrt((B01 + b1)^2 + (B02 + b2)^2)^2 / 4
  var_probe <- 1 - sqrt((pi * zeta_probe) / 2) * exp(-zeta_probe) *
    (besselI(zeta_probe, 0) + besselI(zeta_probe, 1))
  
  results[yr, ] <- c(
    mode_est(circular(aM,modulo="2pi")),
    mean(circular(aM,modulo="2pi")),
    sd(aM),
    hpd_est(aM),
    mode_est(var_probe),
    mean(var_probe),
    sd(var_probe),
    hpd_est(var_probe)
  )
}

results$DOY<- radians_to_days(results$mean_angle%% (2*pi))
results$pheno_year<- c("2018-2019","2019-2020","2020-2021","2021-2022","2022-2023","2023-2024")
View(results)


windows()
ggplot() +
  geom_point(data = hura_mean,aes(x = theta_rad, y = 1), 
    size = 2, alpha = 0.6
  ) +
  coord_polar(start = 0, direction = 1) + 
  geom_segment(data = grand_mean,  
    aes(x = mean, y = 0, xend = mean, yend = 2), 
    arrow = arrow(length = unit(0.2, "cm")), 
    size = 1
  ) +
  geom_segment(data = grand_indv_model_tbl,aes(x = mean, y = 0, xend = mean, yend = 1.5, color=term), 
               arrow = arrow(length = unit(0.2, "cm")) , 
               size = 1)+
  geom_segment(
    data = results,
    aes(x = mean_angle, y = 0, xend = mean_angle, yend = 1.75+mean_var), 
    size = 1
  ) +
geom_point(data = results,
  aes(x = mean_angle, y = 1.75+mean_var,shape = pheno_year), 
  size = 3, stroke = 1.2
)+
  scale_x_continuous(limits = c(0, 2 * pi),
    breaks = 2 * pi * (cumsum(c(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30)) + 15) / 365,
    labels = month.abb
  ) +
  labs(x = NULL, y = NULL,
    title = paste("Mean Day of Year and Mean Resultant Length (", sp, ")", sep = "")
  )


## now lets look at the random effects posteriors

#i dont want the mean of a mean, i should be more robust. what can i use to go from circular.ri to a measure of intraspecific variability 

n_iter <- nrow(fit_leafing$circular.ri)
results_df <- data.frame(
  iteration = numeric(n_iter),
  circular_variance = numeric(n_iter),
  mean_angle = numeric(n_iter),
  day_of_year = numeric(n_iter)
)

for (i in 1:n_iter) {
  sample_i <- fit_leafing$circular.ri[i, ]
  sample_i_circ <- circular(sample_i)

  results_df$iteration[i] <- i
  results_df$circular_variance[i] <- 1 - rho.circular(sample_i_circ)
  results_df$mean_angle[i] <- mean.circular(sample_i_circ)
  results_df$day_of_year[i] <- radians_to_days(results_df$mean_angle[i]%% (2*pi))
}

View(results_df)



all_vals <- as.vector(fit_leafing$circular.ri)
breaks <- seq(min(all_vals, na.rm = TRUE), max(all_vals, na.rm = TRUE), length.out = 50)
colors <- rainbow(nrow(fit_leafing$circular.ri))

windows()
hist(fit_leafing$circular.ri[1, ], breaks = breaks, 
     col = adjustcolor(colors[1], alpha.f = 0.1),
     xlim = c(min(all_vals), max(all_vals)), border = NA,
     main = "Overlapping Histograms of Circular RIs (Hura crepitans)",
     xlab = "Day of Year (converted from radians)", freq = FALSE,
     xaxt = "n")

# Add all remaining histograms with transparency
for (i in 2:nrow(fit_leafing$circular.ri)) {
  hist(fit_leafing$circular.ri[i, ], breaks = breaks, 
       col = adjustcolor(colors[i], alpha.f = 0.1),
       border = NA, add = TRUE, freq = FALSE)
}
radian_ticks <- seq(min(all_vals), max(all_vals), length.out = 20)
doy_labels <- round(radians_to_days(radian_ticks %% (2 * pi)))
axis(side = 1, at = radian_ticks, labels = doy_labels)
abline(v = results_df$mean_angle, col = "black", lwd = 1, lty = 1)
doy_list_text <- paste0(sort(unique(round(results_df$day_of_year))), collapse = ", ")
mtext(paste("Posterior Mean DOYs:", doy_list_text), side = 1, line = 3, cex = 0.8)



df_long <- as.data.frame(t(fit_leafing$circular.ri)) %>%
  mutate(iter = row_number()) %>%
  pivot_longer(-iter, names_to = "tree", values_to = "radian") %>%
  mutate(tree = factor(tree),
         doy = round(radians_to_days(radian %% (2 * pi))))

# Get all radian values for breaks
all_vals <- df_long$radian
breaks <- pretty(all_vals, n = 30)

# Get results_df if not defined
# Assume it has: mean_angle (radians), day_of_year
# results_df <- data.frame(mean_angle = ..., day_of_year = ...)

# Plot
windows()
ggplot(df_long, aes(x = radian, group = tree)) +
  geom_histogram(aes(y = after_stat(density), fill = tree),
                 bins = length(breaks) - 1,
                 position = "identity", alpha = 0.1, color = NA) +
  geom_vline(data = results_df,
             aes(xintercept = mean_angle),
             color = "black", linetype = "solid", linewidth = 0.4) +
  scale_x_continuous(
    name = "Day of Year (converted from radians)",
    breaks = seq(min(all_vals), max(all_vals), length.out = 20),
    labels = round(radians_to_days(seq(min(all_vals), max(all_vals), length.out = 20) %% (2 * pi)))
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(color = "black", face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    plot.title= element_text(size=19),
    plot.margin = margin(10, 10, 40, 10),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12)
  ) +
  labs(
    title = "Overlapping Histograms of Circular RIs (Dipteryx oleifera)",
    y = "Density",
    caption = paste("Random intercepts DOYs:", 
                    paste0(sort(unique(round(results_df$day_of_year))), collapse = ", "))
  )












windows()
#breaks <- seq(-0.01, 0.01, length.out = 200)
hist(intraspecific_var_post, breaks=50)

summary_stats <- c(
  mean = mean(intraspecific_var_post),
  sd = sd(intraspecific_var_post),
  mode = mode_est(intraspecific_var_post),
  HPD = hpd_est(intraspecific_var_post)
)

summary_stats










## addint the globalID as a fixed effect and leaf drop cluster as a random effect

fit_leafing_global <- bpnme(pred.I = theta_rad ~ GlobalID + (1 | leaf_drop_cluster_num),
                            data = hura_mean,
                            its = 2000,
                            burn = 200,
                            n.lag = 5
)

M <- as.matrix(fit_leafing_global$beta1)
View(M)
pheno_year <- colnames(M)[1:15]

results <- data.frame(
  row.names = pheno_year,
  mode_angle = numeric(length(pheno_year)),
  mean_angle = numeric(length(pheno_year)),
  sd_angle = numeric(length(pheno_year)),
  LB_angle = numeric(length(pheno_year)),
  UB_angle = numeric(length(pheno_year)),
  mode_var = numeric(length(pheno_year)),
  mean_var = numeric(length(pheno_year)),
  sd_var = numeric(length(pheno_year)),
  LB_var = numeric(length(pheno_year)),
  UB_var = numeric(length(pheno_year))  
)
B01 <- fit_leafing_global$beta1[, pheno_year[1]]
B02 <- fit_leafing_global$beta2[, pheno_year[1]]


for (i in seq_along(pheno_year)) {
  yr <- pheno_year[i]
  
  b1 <- fit_leafing_global$beta1[, yr]
  b2 <- fit_leafing_global$beta2[, yr]
  
  # Circular mean angle (posterior samples)
  aM <- atan2(B02 + b2, B01 + b1)
  
  # Circular variance
  zeta_probe <- sqrt((B01 + b1)^2 + (B02 + b2)^2)^2 / 4
  var_probe <- 1 - sqrt((pi * zeta_probe) / 2) * exp(-zeta_probe) *
    (besselI(zeta_probe, 0) + besselI(zeta_probe, 1))
  
  results[yr, ] <- c(
    mode_est(aM),
    mean(aM),
    sd(aM),
    hpd_est(aM),
    mode_est(var_probe),
    mean(var_probe),
    sd(var_probe),
    hpd_est(var_probe)
  )
}
View(results)
results$DOY<- radians_to_days(results$mode_angle)%% (2*pi)

###########Frequentist plot using circular package############
windows()
mean_vectors <- hura_mean %>%
  group_by(leaf_drop_cluster) %>%
  summarise(
    mean_length = rho.circular(circular(theta_rad, units = "radians", modulo = "2pi"), na.rm = TRUE),
    mean_angle = mean(circular(theta_rad, units = "radians", modulo = "2pi"), na.rm = TRUE)
  )

mean_vectors$DOY <- radians_to_days(mean_vectors$mean_angle %% (2 * pi))
ggplot() +
  geom_point(data = hura_mean, 
             aes(x = theta_rad, y = 1), 
             size = 2, alpha = 0.6) +
  geom_segment(data = mean_vectors,
               aes(x = mean_angle, y = 0, xend = mean_angle, yend = mean_length), 
               arrow = arrow(length = unit(0.2, "cm")), 
               size = 1) +
  geom_segment(data = fit_leafing_means,
               aes(x = mean, y = 0, xend = mean, yend = 1), 
               arrow = arrow(length = unit(0.2, "cm")),color= "red", 
               size = 1) +
  coord_polar(start = 0, direction = 1) +
  scale_x_continuous(
    limits = c(0, 2 * pi),
    breaks = 2 * pi * (cumsum(c(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30)) + 15) / 365,
    labels = month.abb
  ) +
  facet_wrap(~leaf_drop_cluster) +
  theme_minimal() +
  labs(x = NULL, y = NULL)+
  labs(title = paste("Mean Day of Year and Mean Resultant Length (", sp, ")", sep = ""))
###########################################################
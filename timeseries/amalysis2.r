library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan, lme4, circular, bpnreg,tidyr, tidyverse, bpnreg)

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

sp="Hura crepitans"
hura_clustered <- all %>%
  filter(break_type == "start_leaf_drop")%>%
  filter(latin==sp)

unique(hura_clustered$GlobalID)
#cluster the leaf drop breakpoints
db <- dbscan::dbscan(as.matrix(hura_clustered$date_num), eps = 40, minPts = 10)  ##140 for cavallinesia, 40 for hura 
hura_clustered$leaf_drop_cluster <- db$cluster

# check the clustering
windows()
hura_clustered%>%ggplot(aes(x = date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)


##drop the noise breakpoints
hura_clustered<- hura_clustered%>%  ##no need to remove for cavallinesia
  filter(leaf_drop_cluster!= 7)%>%
  filter(leaf_drop_cluster!= 0)   ##hura has erroneous cluster==7

#More than 2 breakpoint per phenological year should be average between them
base_date <- as.Date("2018-04-04")
hura_mean <- hura_clustered %>% 
  group_by(GlobalID, leaf_drop_cluster) %>% 
  summarise(
    date_num_mean = mean(date_num, na.rm = TRUE),
    .groups = "drop"
  ) %>% 
  mutate(
    mean_date  = as.Date(base_date + round(date_num_mean)),
    year = as.integer(format(mean_date, "%Y")),
    is_leap = (year %% 4 == 0 & year %% 100 != 0) | (year %% 400 == 0),
    dayOfYear  = yday(mean_date),
    days_in_year = ifelse(is_leap, 366, 365),
    theta_rad = dayOfYear * 2 * pi / days_in_year,
  )

hura_mean$leaf_drop_cluster <- recode(hura_mean$leaf_drop_cluster,
                                      `6` = "2018-2019",
                                      `1` = "2019-2020",
                                      `2` = "2020-2021",
                                      `3` = "2021-2022",
                                      `4` = "2022-2023",
                                      `5` = "2023-2024")

hura_mean$leaf_drop_cluster <- recode(hura_mean$leaf_drop_cluster,
                                      `1` = "2018-2019",
                                      `2` = "2019-2020",
                                      `3` = "2020-2021",
                                      `4` = "2021-2022",
                                      `5` = "2022-2023",
                                      `6` = "2023-2024")


windows()
hura_mean%>%ggplot(aes(x = mean_date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)

View(hura_mean)
hura_mean<- hura_mean %>% filter(GlobalID !="87757645-6a6c-41c0-b862-6b192c82f4cb") #this is a cava
hura_mean<- hura_mean %>% filter(GlobalID !="8b94174f-b363-4fda-b5b6-7a76b0a7e51e") #first hura, doesnt event drop leaves
hura_mean<- hura_mean %>% filter(GlobalID !="18cb1282-1df7-42fb-968a-5bbd8b009c8a") #second hura, bad segmentation follow
hura_mean<- hura_mean %>% filter(GlobalID !="1bb104f9-9cfd-4761-bd2c-0e326ac68f0e") #third hura, bad segmentation follow and doesnt loose leaves, maybe lianas
hura_mean<- hura_mean %>% filter(GlobalID !="259620fd-6243-4f43-95fb-cf02edadb818") #fourth hura, bad segmentation follow and doesnt loose leaves, maybe lianas



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
  its = 2000,
  burn = 200,
  n.lag = 5
)

grand_indv_model_tbl <- as_tibble(grand_indv_model$circ.coef.means, rownames = "term")
grand_indv_model_tbl$DOY <- radians_to_days(grand_indv_model_tbl$mean %% (2 * pi))
View(grand_indv_model_tbl)

grand_indv_model_tbl<- grand_indv_model_tbl[1:15,]
grand_indv_model_tbl$mean<- circular(grand_indv_model_tbl$mean, units="radians", modulo="2pi")

###### MODEL WITHIN LEAF DROP CLUSTER AS FIXED EFFECT#########
grand_leaf_model<- bpnr(
  pred.I= theta_rad ~leaf_drop_cluster,
  data = hura_mean,
  its = 2000,
  burn = 200,
  n.lag = 5
)


grand_leaf_model_tbl<- as_tibble(grand_leaf_model$circ.coef.means, rownames= "term")
grand_leaf_model_tbl$DOY<- radians_to_days(grand_leaf_model_tbl$mean %% (2*pi))
grand_leaf_model_tbl$mean<- circular(grand_leaf_model_tbl$mean, units='radians', modulo="2pi")
grand_leaf_model_tbl<- grand_leaf_model_tbl[1:6,]
View(grand_leaf_model_tbl)

###plot for the exploratory models#############################
windows()
grand_mean <- data.frame(mean = atan2(pull(lin_grand_model, mean)[2], pull(lin_grand_model, mean)[1])%% (2*pi))
grand_mean$mean<- circular(grand_mean$mean, units = "radians", modulo = "2pi")

unique(hura_mean$GlobalID)
windows()
ggplot() +
  geom_point(data = hura_mean[hura_mean$GlobalID=="24a5f646-9ef6-4e62-89fe-963cd03a3d0f",],aes(x = theta_rad, y = 1), 
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
    data = grand_leaf_model_tbl,
    aes(x = mean, y = 0, xend = mean, yend = 1.75), 
    size = 1
  ) +
geom_point(data = grand_leaf_model_tbl,
  aes(x = mean, y = 1.75,shape = term), 
  size = 3, stroke = 1.2
)+
  scale_x_continuous(limits = c(0, 2 * pi),
    breaks = 2 * pi * (cumsum(c(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30)) + 15) / 365,
    labels = month.abb
  ) +
  labs(x = NULL, y = NULL,
    title = paste("Mean Day of Year and Mean Resultant Length (", sp, ")", sep = "")
  )

### within species variation in leaf drop timing
var_inv_model <- bpnme(
  pred.I = theta_rad ~ (1 | GlobalID_num),  ###echeck the default prior 
  data = hura_mean,
  its = 2000,
  burn = 200,
  n.lag = 5
)

var_inv_model_tbl <- as_tibble(coef_lin(var_inv_model), rownames = "term")
radians_to_days(atan2(pull(var_inv_model_tbl, mean)[2], pull(var_inv_model_tbl, mean)[1]) %% (2 * pi)) ## same grand mean as grand model

var_inv_model   #115 and 1.21  DIC and param
radians_to_days(0.01689)  ##not significant 

### within cluster variation in leaf drop timing
variance_model <- bpnme(pred.I = theta_rad ~  (1 | leaf_drop_cluster_num) ,
              data = hura_mean,
              its = 2000,
              burn = 200,
              n.lag = 5
)


variance_model   ##71 and 3 DIC and param
radians_to_days(0.0108) # not significant
variance_model_lin<- as_tibble(coef_lin(variance_model),rownames="term")
radians_to_days(atan2(pull(variance_model_lin, mean)[2], pull(variance_model_lin, mean)[1]) %% (2 * pi)) ## same grand mean as grand model


### adding the leaf drop cluster as a fixed effect
fit_leafing <- bpnme(pred.I = theta_rad ~ leaf_drop_cluster + (1 | GlobalID_num),
                    data = hura_mean,
                    its = 10000,
                    burn = 700,
                    n.lag = 5
)

traceplot(fit_leafing, parameter= "beta2")

fit_leafing   #DIC      71   9.26669 param
fit_leafing_linear<- as_tibble(coef_lin(fit_leafing), rownames = "term")
radians_to_days(atan2(pull(fit_leafing_linear, mean)[7], pull(fit_leafing_linear, mean)[1]) %% (2 * pi)) ## 324  so the intercept now is the mean of 2018-2019

fit_leafing_means<- as_tibble(fit_leafing$circ.coef.means, rownames= "term")
fit_leafing_means$DOY<- radians_to_days(fit_leafing_means$mean)%% (2*pi)
fit_leafing_means$mean<- circular(fit_leafing_means$mean, units="radians", modulo="2pi")
fit_leafing_means<- fit_leafing_means[1:6,]
fit_leafing_means$leaf_drop_cluster <- as.factor(c("2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"))

View(fit_leafing_means)


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
View(summary_all_pairs)
colnames(summary_all_pairs)
summary_all_pairs$significance <- ifelse(
  (summary_all_pairs$LB > 0 & summary_all_pairs$UB > 0) |
  (summary_all_pairs$LB < 0 & summary_all_pairs$UB < 0),
  "*",
  ""
)
summary_all_pairs$label_text <- paste0(round(summary_all_pairs$DAY, 0), summary_all_pairs$significance)

windows()
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
    title = "Difference Between All Phenological Year Pairs (Cavallinesia planatifolia)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size=10),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18)
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
View(results)
results$pheno_year<- c("2018-2019","2019-2020","2020-2021","2021-2022","2022-2023","2023-2024")


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

mean(fit_leafing$cRI)  # how concentrated is each individual random intercept

windows()
hist(fit_leafing$circular.ri[1, ], breaks = breaks, col = colors[1],
     xlim = c(min(all_vals), max(all_vals)), border = NA,
     main = "Overlapping Histograms of Circular RIs (Cavallinesia platanifolia)",
     xlab = "Day of Year (converted from radians)", freq = FALSE,
     xaxt = "n")  # suppress default axis

# Add all remaining histograms
for (i in 2:nrow(fit_leafing$circular.ri)) {
  hist(fit_leafing$circular.ri[i, ], breaks = breaks, col = colors[i],
       border = NA, add = TRUE, freq = FALSE)
}

# Define ticks in radians (e.g., every 60° = π/3 ≈ 1.05 rad)
radian_ticks <- seq(min(all_vals),max(all_vals), length.out = 20)  # e.g., 0, 60, 120,...360 degrees

# Convert to DOY using your function
doy_labels <- round(radians_to_days(radian_ticks %% (2*pi)))
# Add axis with DOY
axis(side = 1, at = radian_ticks, labels = doy_labels)

individual_means <- apply(fit_leafing$circular.ri, 1, mean_circ)
radians_to_days(individual_means%% (2*pi))#average across all iterations rowise 
mean(radians_to_days(individual_means%% (2*pi))) # this gives me the mean of those  individual means. o whats the mean random intercept between pheno_year

#i dont want the mean of a mean, i should be more robust. what can i use to go from circular.ri to a measure of intraspecific variability 

n_iter <- ncol(fit_leafing$circular.ri)
intraspecific_var_post <- numeric(n_iter)

# For each iteration, compute circular variance of all individual intercepts at that iteration
for (i in 1:n_iter) {
  sample_i <- fit_leafing$circular.ri[, i]
  sample_i_circ <- circular(sample_i)
  intraspecific_var_post[i] <- 1 - rho.circular(sample_i_circ)
}

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
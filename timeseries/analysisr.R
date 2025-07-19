library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan, lme4, circular, bpnreg,tidyr, tidyverse)
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

#lets try a vonmisses distribution for the ceiba
#first remove the interpolated values

month_breaks_regular <- c(1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366)
month_breaks_leap    <- c(1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336, 367)
month_labels <- month.abb

all <- all %>%
  mutate(
    year = as.integer(format(date, "%Y")),
    is_leap = (year %% 4 == 0 & year %% 100 != 0) | (year %% 400 == 0),
    days_in_year = ifelse(is_leap, 366, 365),
    angle_deg = 360 * (dayYear / days_in_year),
    bin = cut(angle_deg, breaks = c(0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365), include.lowest = TRUE, labels = month.abb)
  )

# Step 2: Plot boxplot with coord_polar()
# Create circular tick labels
y_ticks <- seq(0, 75, by = 25)
y_labels <- data.frame(
  x = rep(1, length(y_ticks)),  # Position at the dummy x value, this will be adjusted later
  y = y_ticks,            # Inverted y values to match reversed scale
  label = as.character(y_ticks)
)

all$bin

ggplot(all, aes(x = bin, y = leafing)) +
  geom_boxplot(outlier.alpha = 0.1, fill = "Goldenrod", color = "black", width = 0.8) +
  coord_polar(start = 0, direction = 1) +
  scale_y_reverse(limits = c(100, 0)) +
  facet_wrap(~ latin) +  
  theme_minimal(base_size = 14) +  # Optional: globally increase text size
  labs(title = "Leaf Coverage Patterns Across Tree Species", x = "", y = "") +
  labs(subtitle= "Radial boxplots show 7 years of leaf coverage data binned by month")+
  theme(
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 14, face = "bold"),         # Thicker (larger, bold) x-axis labels
    panel.grid.major = element_line(size = 1),                    # Thicker major grid lines
    panel.grid.minor = element_line(size = 0.6),                   # Optional: thicker minor grid too
    strip.text = element_text(size = 16, face = "bold") 
  )+
  geom_text(data = y_labels, aes(x = x, y = y, label = label), hjust = 1, size = 4, angle = 0)


##focusing on Hura crepitans and start of leaf drop
hura_clustered <- all %>%
  filter(break_type == "start_leaf_drop")%>%
  filter(latin=="Hura crepitans")


#cluster the leaf drop breakpoints
db <- dbscan::dbscan(as.matrix(hura_clustered$date_num), eps = 40, minPts = 10)
hura_clustered$leaf_drop_cluster <- db$cluster

##drop the noise breakpoints
hura_clustered<- hura_clustered%>%
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
                                      `1` = "2019-2020",
                                      `2` = "2020-2021",
                                      `3` = "2021-2022",
                                      `4` = "2022-2023",
                                      `5` = "2023-2024",
                                      `6` = "2018-2019")

windows()
hura_mean%>%ggplot(aes(x = mean_date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)




#to transform this shit to the linear space i need to find the min and max dates of all seasons
# it is circular, we need to go to the ciruclar space

hura_range <- hura_mean %>%
  group_by(leaf_drop_cluster) %>%
  summarise(
    sorted_theta = list(sort(circular(theta_rad, units = "radians", modulo = "2pi"))),
    .groups = "drop"
  ) %>%
  mutate(
    arc_bounds = map(sorted_theta, ~{
      th <- as.numeric(.x)
      diffs <- diff(c(th, th[1] + 2*pi))  # wraparound gap
      gap_idx <- which.max(diffs)
      min_theta <- th[(gap_idx %% length(th)) + 1]
      max_theta <- th[gap_idx]
      tibble(min_theta = min_theta, max_theta = max_theta)
    })
  ) %>%
  unnest(arc_bounds) %>%
  mutate(
    min_DOY = round(min_theta / (2 * pi) * 365),
    max_DOY = round(max_theta / (2 * pi) * 365),
    min_DOY = ifelse(min_DOY == 0, 1, min_DOY),
    max_DOY = ifelse(max_DOY == 0, 1, max_DOY)
  )

print(hura_range)
mindate<-min(hura_range$min_DOY)-1  ## This is day 0
maxdate<-max(hura_range$max_DOY)-1

# Now we will transform each leaf drop cluster to the linear space
hura_mean <- hura_mean %>%
  mutate(
    DOYlinear = ##every mean_date is days since mindate
      ifelse(
        dayOfYear < mindate,
        dayOfYear + 365 - mindate,  # wrap around to the end of the year
        dayOfYear - mindate
      ))

View(hura_mean)

##good lets visualize DOYlinear per leaf drop cluster
windows()
ggplot(hura_mean, aes(x = DOYlinear, y = leaf_drop_cluster, color = factor(GlobalID))) +
  geom_jitter(height = 0.1, alpha = 0.7, size = 2) +
  labs(title = "Leaf Drop Clusters in Linear Space (Hura crepitans)",
       x = "Days Since Minimum Date",
       y = "GlobalID",
       color = "Leaf Drop Cluster") +
  theme_minimal(base_size = 14)+
  theme(legend.position = "none")


## now we can fit a linear to DOYlinear using a Bayesian approach

library(brms)

hura_mean$leaf_drop_cluster <- factor(hura_mean$leaf_drop_cluster)
model <- brm(
  DOYlinear ~ leaf_drop_cluster + (1 | GlobalID),
  data = hura_mean,
  family = gaussian(),
  chains = 4,
  iter = 2000,
  cores = 4,
  seed = 123
)

summary(model)


fixed_table <- round(as.data.frame(fixef(model)), 2)
windows(width = 10, height = 6)
grid.table(fixed_table)

ranef(model)
random_table <- round(as.data.frame(ranef(model)$GlobalID), 2)
View(random_table)
windows(width = 10, height = 6)
grid.table(random_table)

library(gridExtra)
re_sd <- as.data.frame(VarCorr(model))
windows(width = 10, height = 6)
grid.table(re_sd)

windows()
conditional_effects(model, effects = "leaf_drop_cluster")


## model 2 aims to partition the variance of DOYlinear into two components: one for the leaf drop cluster and one for the GlobalID
# This allows us to see how much of the variance in DOYlinear is explained by the
model2 <- brm(
  DOYlinear ~ 1 + (1 | leaf_drop_cluster) + (1 | GlobalID),
  data = hura_mean,
  family = gaussian(),
  chains = 4,
  iter = 2000,
  cores = 4,
  seed = 123
)

summary(model2)


##pull the estimate of each random effect 
# Use regular year breaks


windows()
mean_vectors <- hura_mean %>%
  group_by(leaf_drop_cluster) %>%
  summarise(
    mean_length = rho.circular(circular(theta_rad, units = "radians", modulo = "2pi"), na.rm = TRUE),
    mean_angle = mean(circular(theta_rad, units = "radians", modulo = "2pi"), na.rm = TRUE)
  )
ggplot() +
  geom_point(data = hura_mean, 
             aes(x = theta_rad, y = 1), 
             size = 2, alpha = 0.6) +
  geom_segment(data = mean_vectors,
               aes(x = mean_angle, y = 0, xend = mean_angle, yend = mean_length), 
               arrow = arrow(length = unit(0.2, "cm")), 
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
  labs(title = "Mean Day of Year and Mean Resultant Length (Hura crepitans)")




library(bpnreg)
hura_mean$GlobalID_num <- as.numeric(as.factor(hura_mean$GlobalID))
fit_leafing <- bpnme(
  pred.I = theta_rad ~ leaf_drop_cluster,,
  data = hura_mean,
  its = 10000,
  burn = 2000,
  n.lag = 5
)
































fit_leafing$circ.coef.cat    ##the difference two phenological years
cat_tbl <- as_tibble(fit_leafing$circ.coef.cat, rownames = "term")
View(cat_tbl)

cat_tbl$difDOY <- cat_tbl$mean * 365 / (2 * pi)
View(cat_tbl)

table2 <- cat_tbl[1:5, ]
table2[ , 2:7] <- round(table2[ , 2:7], 2)  # Assuming columns 2 to 7 are numeric
windows(width = 10, height = 6)
grid.table(table2)



install.packages("tidyverse")
hura_end <- all %>%
  filter(break_type == "end_leaf_flush")%>%
  filter(latin=="Hura crepitans")


# DBSCAN clustering
db <- dbscan::dbscan(as.matrix(hura_clustered$date_num), eps = 40, minPts = 10)
hura_clustered$leaf_drop_cluster <- db$cluster

db2 <- dbscan::dbscan(as.matrix(hura_end$date_num), eps = 40, minPts = 10)
hura_end$leaf_drop_cluster <- db2$cluster


##drop the noise breakpoints
hura_clustered<- hura_clustered%>%
  filter(leaf_drop_cluster!= 7)%>%
  filter(leaf_drop_cluster!= 0)   ##hura has erroneous cluster==7

hura_end<- hura_end%>%
  filter(leaf_drop_cluster!= 6)%>%
  filter(leaf_drop_cluster!= 0)


#easy, if two break points belong to the same cluster we are going to average it
base_date <- as.Date("2018-04-04")

hura_mean <- hura_clustered %>% 
  group_by(GlobalID, leaf_drop_cluster) %>% 
  summarise(
    date_num_mean = mean(date_num, na.rm = TRUE),
    .groups = "drop"
  ) %>% 
  mutate(
    mean_date  = as.Date(base_date + round(date_num_mean)),
    dayOfYear  = yday(mean_date),
    theta_rad = dayOfYear * 2 * pi / 365,
    sin_theta = sin(theta_rad),
    cos_theta = cos(theta_rad)
  )
windows()
hura_mean%>%ggplot(aes(x = mean_date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)


hura_end_mean <- hura_end%>% 
  group_by(GlobalID, leaf_drop_cluster) %>% 
  summarise(
    date_num_mean = mean(date_num, na.rm = TRUE),
    .groups = "drop"
  ) %>% 
  mutate(
    mean_date  = as.Date(base_date + round(date_num_mean)),
    dayOfYear  = yday(mean_date),
    theta_rad = dayOfYear * 2 * pi / 365,
    sin_theta = sin(theta_rad),
    cos_theta = cos(theta_rad)
  )
windows()
hura_end_mean%>%ggplot(aes(x = mean_date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)

hura_mean$leaf_drop_cluster <- recode(hura_mean$leaf_drop_cluster,
                                      `1` = "2019-2020",
                                      `2` = "2020-2021",
                                      `3` = "2021-2022",
                                      `4` = "2022-2023",
                                      `5` = "2023-2024",
                                      `6` = "2018-2019")

hura_end_mean$leaf_drop_cluster <- recode(hura_end_mean$leaf_drop_cluster,
                                      `1` = "2019-2020",
                                      `2` = "2020-2021",
                                      `3` = "2021-2022",
                                      `4` = "2022-2023",
                                      `5` = "2018-2019")

#the random effects must be numerics 
hura_mean$GlobalID_num <- as.numeric(as.factor(hura_mean$GlobalID))
hura_end_mean$GlobalID_num <- as.numeric(as.factor(hura_end_mean$GlobalID))

fit_leafing <- bpnme(
  pred.I = theta_rad ~ leaf_drop_cluster + (1 | GlobalID_num),
  data = hura_mean,
  its = 10000,
  burn = 2000,
  n.lag = 5
)
print(fit_leafing)

#visualize first component
windows()
traceplot(fit_leafing, "beta1")
windows()
traceplot(fit_leafing, "beta2")

## you can print the full estimate table
fit_leafing$circ.coef.cat    ##the difference two phenological years
cat_tbl <- as_tibble(fit_leafing$circ.coef.cat, rownames = "term")
View(cat_tbl)

fit_leafing$circ.coef.means  ## the mean coefficients for each cluster
coef_tbl <- as_tibble(fit_leafing$circ.coef.means, rownames = "term")
View(coef_tbl)

fit_leafing$circ.res.varrand  ## residual variance? The mean, mode, standard deviation and 95 random intercepts and slopes.
varrand_tbl <- as_tibble(fit_leafing$circ.res.varrand, rownames = "term")
View(varrand_tbl)


circular_ri <- fit_leafing$circular.ri  # Matrix: individuals x iterations ## A vector of posterior samples for the circular random intercepts.

# Calculate mean resultant length (MRL) across individuals for each iteration
mrl_per_iteration <- apply(circular_ri, 2, function(theta) {
  rho.circular(circular(theta, units = "radians"))
})
mean_cvar <- mean(1 - mrl_per_iteration)

# Plot posterior density of circular variance
ggplot(data.frame(cvar = 1 - mrl_per_iteration), aes(x = cvar)) +
  geom_density(fill = "lightblue", alpha = 0.6) +
  geom_vline(xintercept = quantile(1 - mrl_per_iteration, c(0.025, 0.975)), linetype = "dashed", color = "blue") +
  geom_vline(xintercept = mean_cvar, color = "black") +
  labs(title = "Posterior Distribution of Circular Random Intercept Variance",
       x = "Circular Variance (1 - MRL)",
       y = "Density") +
  theme_minimal()






fit_end <- bpnme(
  pred.I = theta_rad ~ leaf_drop_cluster + (1 | GlobalID_num),
  data = hura_end_mean,
  its = 10000,
  burn = 2000,
  n.lag = 5
)



print(fit_leafing$circ.res.varrand)

# Compare mean of 1 - mrl_per_iteration to circ.res.varrand mean for RI


coef_tbl <- as_tibble(fit_leafing$circ.coef.means, rownames = "term")
coef_tbl_end <- as_tibble(fit_end$circ.coef.means, rownames = "term")

intercept_end <- coef_tbl |> 
  filter(term == "(Intercept)") |> 
  pull(mean)

coef_tbl_end <- coef_tbl_end |>
  mutate(
    est = if_else(term == "(Intercept)", mean, intercept_end + mean),
    meanDOY = (est %% (2 * pi)) * (365 / (2 * pi))
  )

intercept_effect <- coef_tbl_plot_end$effect[coef_tbl_plot_end$term == "(Intercept)"]
coef_tbl_plot_end$effect <- coef_tbl_plot_end$effect - intercept_effect



intercept <- coef_tbl |> 
  filter(term == "(Intercept)") |> 
  pull(mean)

coef_tbl <- coef_tbl |>
  mutate(
    est = if_else(term == "(Intercept)", mean, intercept + mean),
    meanDOY = (est %% (2 * pi)) * (365 / (2 * pi))
  )


arrow_length <- 0.9  # arrow length
coef_tbl_plot <- coef_tbl %>% 
  slice(1:6) %>%
  mutate(
    angle = est %% (2 * pi),  # ensure angles 0 to 2pi
    radius_start = 0,
    radius_end = arrow_length
  )

View(coef_tbl_plot)
coef_tbl_plot$effect<-round((coef_tbl_plot$mean * 365) / (2 * pi), 1)


coef_tbl_plot$type<-"Start of Leaf Drop"
coef_tbl_plot_end <- coef_tbl_end %>% 
  slice(1:5) %>%
  mutate(
    angle = est %% (2 * pi),  # ensure angles 0 to 2pi
    radius_start = 0,
    radius_end = arrow_length
  )
coef_tbl_plot_end$type<-"End of Leaf Flush"

coef_tbl_plot_end$effect<-round((coef_tbl_plot_end$mean * 365) / (2 * pi), 1)
View(coef_tbl_plot_end)
combined<- bind_rows(coef_tbl_plot, coef_tbl_plot_end)
View(combined)

combined$effect <- round((combined$mean * 365) / (2 * pi), 1)
coef_tbl_plot
months <- tibble(
  month = month.abb,
  angle = seq(0, 2 * pi - 2 * pi / 12, length.out = 12),
  x = seq(0, 330, by = 30),  # degrees for scale_x_continuous breaks
  y = rep(arrow_length + 0.1, 12) # just outside arrows
)

season_labels <- c(
  "(Intercept)" = "2018-2019",
  "leaf_drop_cluster2019-2020" = "2019–2020",
  "leaf_drop_cluster2020-2021" = "2020–2021",
  "leaf_drop_cluster2021-2022" = "2021–2022",
  "leaf_drop_cluster2022-2023" = "2022–2023",
  "leaf_drop_cluster2023-2024" = "2023–2024"
)

season_colors <- c(
  "(Intercept)" = "black",
  "leaf_drop_cluster2019-2020" = "#1b9e77",
  "leaf_drop_cluster2020-2021" = "#d95f02",
  "leaf_drop_cluster2021-2022" = "#7570b3",
  "leaf_drop_cluster2022-2023" = "#e7298a",
  "leaf_drop_cluster2023-2024" = "#66a61e"
)

ggplot(combined) +
  # Arrows for both types
  geom_segment(aes(x = angle, xend = angle, y = radius_start, yend = radius_end,
                   color = term, linetype = type),
               arrow = arrow(length = unit(0.15, "inches")), size = 1.2) +

  coord_polar(start = 0, direction = 1) +

  scale_x_continuous(
    limits = c(0, 2 * pi),
    breaks = seq(0, 2 * pi - 2 * pi / 12, length.out = 12),
    labels = month.abb
  ) +
  scale_color_manual(values = season_colors, labels = season_labels, name = "Season")+
  scale_linetype_manual(values = c(
    "Start of Leaf Drop" = "solid",
    "End of Leaf Flush" = "dashed"
  )) +
    theme_minimal(base_size = 14) +
  theme(
    axis.title = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    panel.grid.major = element_line(size = 1),
    panel.grid.minor = element_line(size = 0.6),
    legend.position = "right"
  ) +
  labs(
    title = "Circular Timing of Leaf Phenology (Hura crepitans)",
    subtitle = "Model-adjusted to account for intraspecific variation",
    color = "Season",
    linetype = "Timing event"
  )



#Awesome on to the next question.










hura_mean$leaf_drop_cluster_num<- as.numeric(as.factor(hura_mean$leaf_drop_cluster))
hura_mean$GlobalID<-as.factor(hura_mean$GlobalID)

fit_intra <- bpnme(
  pred.I = theta_rad ~ GlobalID + (1|leaf_drop_cluster_num),
  data = hura_mean,
  its = 10000,
  burn = 2000,
  n.lag = 5
)

temp<- as_tibble(fit_intra$circ.coef.means, rownames = "term")
temp2<- as_tibble(fit_intra$circ.coef.cat, rownames= "term")
merged <- temp %>% left_join(temp2, by = "term")


radians_to_days <- function(r) {
  (r / (2 * pi)) * 365
}

radians_to_DOY <- function(r) {
  doy <- (r %% (2 * pi)) / (2 * pi) * 365
  # Wrap 0 to 365
  doy[doy == 0] <- 365
  return(doy)
}

flag_significance_direction <- function(lb, ub) {
  if (lb > 0 & ub > 0) {
    return("significant")
  } else if (lb < 0 & ub < 0) {
    return("significant")
  } else {
    return("not significant")
  }
}

merged <- merged %>%
  mutate(
    DOY_relative = radians_to_days(mean.y),     # Relative difference from reference
    DOY_absolute = radians_to_DOY(mean.x)       # Wrapped absolute DOY for calendar scale
  )

merged <- merged %>%
  mutate(DOY_relative = -DOY_relative)

merged <- merged %>%
  mutate(
    significance = NA_character_  # Initialize empty column
  )

merged[-1, ] <- merged[-1, ] %>%
  rowwise() %>%
  mutate(significance = flag_significance_direction(LB.y, UB.y)) %>%
  ungroup()

summary_table <- merged[2:61, ] %>%
  count(significance) %>%
  mutate(
    percent = round(100 * n / sum(n), 1),
    label = case_when(
      significance == "significant" ~ "Significantly different",
      significance == "not significant" ~ "Not significantly different"
    )
  ) %>%
  select(`Timing Difference` = label, `Individuals (n)` = n, `Percentage (%)` = percent)

# Create table figure
fig_table <- ggtexttable(summary_table, rows = NULL, theme = ttheme("light"))
# Display it
fig_table



plot_data <- merged[2:61, ]  # Only individuals

View(plot_data)

# Optional: shorten the GlobalID or make a new ID label
plot_data <- plot_data %>%
  mutate(label = substr(term, 1, 15))  # Shorten if needed

# Plot
ggplot(plot_data, aes(x = reorder(label, DOY_relative), y = DOY_relative, fill = significance)) +
  geom_col(width = 0.7, color = "black") +  # outline stays black, fill comes from significance
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c(
    "significant" = "green",
    "not significant" = "#d3d3d3"
  )) +
  labs(
    title = "Start of Leaf Drop – Hura crepitans",
    x = "Days Relative to Reference",
    y = "Individual Tree (ID)",
    fill = "Significance"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.y = element_text(size = 8)
  ) +
  coord_flip()



#end of leaf flush
hura_end_mean$leaf_drop_cluster_num<- as.numeric(as.factor(hura_end_mean$leaf_drop_cluster))
hura_end_mean$GlobalID<-as.factor(hura_end_mean$GlobalID)
fit_intra_end <- bpnme(
  pred.I = theta_rad ~ GlobalID + (1 | leaf_drop_cluster_num),
  data = hura_end_mean,
  its = 10000,
  burn = 2000,
  n.lag = 5
)


end<- as_tibble(fit_intra_end$circ.coef.means, rownames = "term")
end2<- as_tibble(fit_intra_end$circ.coef.cat, rownames= "term")
merged_end <- end %>% left_join(end2, by = "term")



merged_end <- merged_end %>%
  mutate(
    DOY_relative = radians_to_days(mean.y),     # Relative difference from reference
    DOY_absolute = radians_to_DOY(mean.x)       # Wrapped absolute DOY for calendar scale
  )

merged_end <- merged_end %>%
  mutate(DOY_relative = -DOY_relative)

merged_end <- merged_end %>%
  mutate(
    significance = NA_character_  # Initialize empty column
  )

merged_end[-1, ] <- merged_end[-1, ] %>%
  rowwise() %>%
  mutate(significance = flag_significance_direction(LB.y, UB.y)) %>%
  ungroup()

summary_table2 <- merged_end[2:61, ] %>%
  count(significance) %>%
  mutate(
    percent = round(100 * n / sum(n), 1),
    label = case_when(
      significance == "significant" ~ "Significantly different",
      significance == "not significant" ~ "Not significantly different"
    )
  ) %>%
  select(`Timing Difference` = label, `Individuals (n)` = n, `Percentage (%)` = percent)

# Create table figure
fig_table <- ggtexttable(summary_table2, rows = NULL, theme = ttheme("light"))
fig_table

plot_data_end <- merged_end[2:61, ]  # Only individuals

# Optional: shorten the GlobalID or make a new ID label
plot_data_end <- plot_data_end%>%
  mutate(label = substr(term, 1, 15))  # Shorten if needed

ggplot(plot_data_end, aes(x = reorder(label, DOY_relative), y = DOY_relative, fill = significance)) +
  geom_col(width = 0.7, color = "black") +  # outline stays black, fill comes from significance
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c(
    "significant" = "green",
    "not significant" = "#d3d3d3"
  )) +
  labs(
    title = "Start of Leaf Drop – Hura crepitans",
    x = "Days Relative to Reference",
    y = "Individual Tree (ID)",
    fill = "Significance"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.y = element_text(size = 8)
  ) +
  coord_flip()



View(hura_mean)
ggplot(hura_mean) +
  # Arrows for both types
  geom_segment(aes(x = theta_rad, xend = theta_rad, y = 0, yend = 0.9,
                   color = leaf_drop_cluster),
               arrow = arrow(length = unit(0.15, "inches")), size = 1.2) +

  coord_polar(start = 0, direction = 1) +

  scale_x_continuous(
    limits = c(0, 2 * pi),
    breaks = seq(0, 2 * pi - 2 * pi / 12, length.out = 12),
    labels = month.abb
  ) +
  scale_linetype_manual(values = c(
    "Start of Leaf Drop" = "solid",
    "End of Leaf Flush" = "dashed"
  )) +
    theme_minimal(base_size = 14) +
  theme(
    axis.title = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    panel.grid.major = element_line(size = 1),
    panel.grid.minor = element_line(size = 0.6),
    legend.position = "right"
  ) +
  labs(
    title = "Circular Timing of Leaf Phenology (Hura crepitans)",
    subtitle = "Start of leaf drop, raw angles",
    color = "Season",
    linetype = "Timing event"
  )



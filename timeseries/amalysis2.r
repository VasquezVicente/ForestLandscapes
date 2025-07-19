library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan, lme4, circular, bpnreg,tidyr, tidyverse, bpnreg)
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

sp="Cavallinesia platanifolia"
hura_clustered <- all %>%
  filter(break_type == "start_leaf_drop")%>%
  filter(latin==sp)

#cluster the leaf drop breakpoints
db <- dbscan::dbscan(as.matrix(hura_clustered$date_num), eps = 140, minPts = 10)  ##140 for cavallinesia, 40 for hura 
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
                                      `2` = "2019-2020",
                                      `3` = "2020-2021",
                                      `4` = "2021-2022",
                                      `5` = "2022-2023",
                                      `6` = "2023-2024",
                                      `1` = "2018-2019")

windows()
hura_mean%>%ggplot(aes(x = mean_date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2)


###########################################
options("contrasts")
options(contrasts = c("contr.treatment", "contr.poly")) # set contrasts to treatment contrasts for categorical variables
options("contrasts")
options(contrasts = c("contr.sum", "contr.poly")) # set contrasts to treatment contrasts for categorical variables

# good lets fit the damn leaf drop model

grand_model<- bpnr(pred.I = theta_rad ~ 1,
                   data = hura_mean,
                   its = 2000,
                   burn = 200,
                   n.lag = 5
)
grand_model
the<- atan2(-0.4393731,1.933337)
radians_to_days(the%% (2 * pi))

hura_mean$leaf_drop_cluster <- as.factor(hura_mean$leaf_drop_cluster) #important
hura_mean$GlobalID_num <- as.numeric(as.factor(hura_mean$GlobalID))
hura_mean$leaf_drop_cluster_num <- as.numeric(as.factor(hura_mean$leaf_drop_cluster))

fit_leafing <- bpnme(
  pred.I = theta_rad ~(1 | GlobalID_num),
  data = hura_mean,
  its = 2000,
  burn = 200,
  n.lag = 5
)
fit_leafing
the<- atan2(-0.4999462,2.159661)
radians_to_days(the%% (2 * pi))


variance_model <- bpnme(pred.I = theta_rad ~  (1 | leaf_drop_cluster_num) ,
              data = hura_mean,
              its = 2000,
              burn = 200,
              n.lag = 5
)

variance_model
the<- atan2(-0.4737049,2.194281)
radians_to_days(the%% (2 * pi))



mean_tbl<-as_tibble(fit_leafing$circ.coef.means,rownames="term")
mean_tbl$DOY<- radians_to_days(mean_tbl$mean %% (2 * pi))
View(mean_tbl)
mean_tbl<- mean_tbl[1:6,]
mean_tbl$mean<- circular(mean_tbl$mean, units = "radians", modulo = "2pi")
mean_tbl$leaf_drop_cluster <- as.factor(c("2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"))
# Convert to DOY
radians_to_days <- function(r) {
  (r / (2 * pi)) * 365
}







radians_to_days(x)
dif_tbl<- as_tibble(fit_leafing$circ.coef.cat,rownames="term")
dif_tbl$DOY<- radians_to_days(dif_tbl$mean %% (2 * pi))
View(dif_tbl)
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
  geom_segment(data = mean_tbl,
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
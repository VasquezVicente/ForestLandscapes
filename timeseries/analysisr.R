library(dplyr)
library(ggplot2)
library(lubridate)
library(dbscan)
hura<- read.csv("C:/Users/VasquezV/repo/ForestLandscapes/timeseries/datset_analysis/hura_analysis.csv")
hura$date <- as.Date(hura$date, format = "%Y-%m-%d")

target_id<- "afbda3c3-9e08-40c8-a682-d6c87c4dd38a"
target<- hura %>% filter(GlobalID== target_id)
ggplot(target, aes(x = date, y = leafing_predicted, color = break_type)) +
  geom_point(size = 2) +
  geom_line(aes(group = 1), color = "gray70", linetype = "dashed") +
  labs(
    title = paste("Leafing over Time for", target_id),
    x = "Date",
    y = "Leafing",
    color = "Break Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )


hura_clustered <- hura %>%
  filter(break_type == "start_leaf_drop")
# DBSCAN clustering
db <- dbscan::dbscan(as.matrix(hura_clustered$date_num), eps = 40, minPts = 10)
hura_clustered$leaf_drop_cluster <- db$cluster


hura_filtered %>%
  filter(break_type == "start_leaf_drop") %>%
  ggplot(aes(x = date, y = GlobalID, color = factor(leaf_drop_cluster))) +  # color by group ID
  geom_jitter(height = 0.1, alpha = 0.7, size = 2) +
  labs(
    title = "Start of Leaf Drop Timeline",
    x = "Date",
    y = "GlobalID",
    color = "Leaf Drop\nEvent ID"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "right"
  )

hura_filtered <- hura_clustered %>%
  filter(leaf_drop_cluster %in% c(1,2,3,4,5,6 )) %>%
  group_by(GlobalID) %>%
  filter(n() >= 4) %>%
  ungroup()



library(lme4)
library(circular)


hura_circ_mean <- hura_filtered %>%
  group_by(GlobalID, leaf_drop_cluster) %>%
  summarise(
    circular_mean_rad = mean(circular(dayYear * 2 * pi / 365, units = "radians")),
    circular_sd_rad = sd.circular(circular(dayYear * 2 * pi / 365, units = "radians")),
    .groups = "drop"
  ) %>%
  mutate(
    circular_mean_day = round((as.numeric(circular_mean_rad) %% (2 * pi)) * 365 / (2 * pi)),
    circular_sd_day = round(as.numeric(circular_sd_rad) * 365 / (2 * pi))
  )


tree_means <- hura_circ_mean %>%
  group_by(GlobalID) %>%
  summarise(
    mean_circular_day = mean(circular(circular_mean_day, units = "degrees", template = "clock24")),
    .groups = "drop"
  ) %>%
  mutate(mean_dayYear = round(as.numeric(mean_circular_day) * 365 / 360))

all <- hura_circ_mean %>%
  left_join(tree_means, by = "GlobalID") %>%
  mutate(
    mean_dayYear = ifelse(mean_dayYear < 0, 365 + mean_dayYear, mean_dayYear)
  )

all <- all %>%
  mutate(
    circular_mean_rad = circular_mean_day * 2 * pi / 365,
    mean_rad = mean_dayYear * 2 * pi / 365,
    angle_diff_rad = atan2(sin(circular_mean_rad - mean_rad), cos(circular_mean_rad - mean_rad)),
    date_centered = angle_diff_rad * 365 / (2 * pi)
  )

all <- all %>%
  mutate(
    leaf_drop_cluster = factor(leaf_drop_cluster,
                               levels = 1:6,
                               labels = c("2021", "2023", "2024", "2019", "2020", "2022")
    )
  )


model <- lm(date_centered ~ leaf_drop_cluster, data = all)
summary(model)

model <- lmer(circular_mean_day ~ leaf_drop_cluster + (1 | GlobalID), data = all)
summary(model)


ggplot(all, aes(x = factor(leaf_drop_cluster), y = circular_mean_day)) +
  geom_boxplot() +
  labs(x = "Leaf Drop Cluster", y = "Mean Day of Year") +
  theme_minimal()


all$DOY_centered_by_tree <- ave(all$circular_mean_day, all$GlobalID, FUN = function(x) x - mean(x))

model <- lmer(DOY_centered_by_tree ~ leaf_drop_cluster + (1 | GlobalID), data = all)
summary(model)






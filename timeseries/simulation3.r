library(pacman)
p_load(dplyr, lubridate, ggplot2,brms,tidyr, modeest)

f_decay <- function(values, k_d, t_d){
    1/ (1 + exp(k_d *(values - t_d)))
}

f_flush <- function(values, k_f, delta, t_d){
    1/ (1 + exp(-k_f * (values - (t_d + delta))))
}

f_phase <- function(values, t_d, delta, values){
    1/ (1 + exp(-0.5 * (values - (t_d + delta/2))))   
}

values1 <- seq(0, 190, by=1)

LC <- function(values, k_d, k_f, t_d, delta) {
  result <- ((1 - f_phase(values, t_d, delta, values)) * f_decay(values, k_d, t_d)) +
        (f_phase(values, t_d, delta, values) * f_flush(values, k_f, delta, t_d))
  return(result)
}


LC_values<- LC(values1, k_d=0.8, k_f=0.8, t_d=30, delta=15)

windows()
plot(values1, LC_values, type='l', col='black', ylim=c(0,1))

form_main <- bf(
  y_norm ~ (1 - (1 / (1 + exp(-0.5 * (day - (td + delta/2)))))) *
             (1 / (1 + exp(kd * (day - td)))) +
           (1 / (1 + exp(-0.5 * (day - (td + delta/2)))) ) *
             (1 / (1 + exp(-kf * (day - (td + delta))))),
  kd + kf + td + delta ~ 1 + pheno_year + (1 | tree),
  nl = TRUE
)




data<- read.csv("timeseries\\dataset_extracted\\cavallinesia.csv")

data<- data %>%
filter(!GlobalID %in% c("87757645-6a6c-41c0-b862-6b192c82f4cb",
                        "6e7eb081-2a27-44a7-85f2-a88a40e3e53a"))

data <- data %>%
  mutate(
    y_norm= pmin(pmax(leafing / 100, 1e-4), 1 - 1e-4),
    date = as.Date(date),
    date_num = as.numeric(difftime(date, as.Date("2018-04-04"), units = "days")),
    DOY= yday(date),
    year= year(date),
    month= month(date),
    pheno_year = if_else(month >= 9, year, year - 1),
    day = as.numeric(difftime(date, as.Date(paste0(pheno_year, "-09-01")), units = "days")),
    tree= as.factor(tag),
    pheno_year= as.factor(pheno_year),
    tree_year= as.factor(paste0(tree, "_", pheno_year))
  )

#maybe reorder the levels of pheno_year
priors <- c(
  prior(normal(110, 8), nlpar = "td", lb = 0),        # centered at 110 days, SD = 4 standard deviation
  prior(normal(170, 10),  nlpar = "delta", lb = 0),     # centered at 50 days, SD = 8 days
  prior(normal(0.6, 0.25), nlpar = "kd", lb = 0),
  prior(normal(0.6, 0.25), nlpar = "kf", lb = 0),
  prior(exponential(0.4), class = "sd", nlpar = "td"),
  prior(exponential(0.4), class = "sd", nlpar = "delta") # mean ≈ 0.67
)
windows()
hist(rnorm(8000,0,10), breaks= 100)

td_norm <- data.frame(
  day = samples <- rnorm(8000, mean = 115, sd = 6)
)
delta_norm <- data.frame(
  duration = samples <- rnorm(8000, mean = 170, sd = 10)
)

td_quantiles <- quantile(td_norm$day, c(0.05, 0.5, 0.95))
delta_quantiles <- quantile(delta_norm$duration, c(0.05, 0.5, 0.95))

segments <- data.frame(
  stat = c("p5", "median", "p95"),
  start = c(td_quantiles[2], td_quantiles[2], td_quantiles[2]),  # 95%, 50%, 5% of td
  end = c(td_quantiles[3] + delta_quantiles[1],    # 95% td + 5% delta
          td_quantiles[2] + delta_quantiles[2],    # 50% td + 50% delta  
          td_quantiles[1] + delta_quantiles[3]),
  y_height= c(0.6, 0.5, 0.4)    # 5% td + 95% delta
)
print(segments)

windows()
ggplot(data, aes(x=day, y=y_norm, color=tree))+
  geom_line()+
  geom_histogram(data= td_norm, aes(x= day, y= (..density..)*10), bins= 100, inherit.aes = FALSE, alpha= 0.9, fill= "gray")+
  geom_segment(data= segments, aes(x= start, xend= end, y= y_height, yend= y_height, color= stat), size= 2)+
  facet_wrap(~pheno_year)+
  theme_minimal()+
  scale_x_continuous(breaks= seq(0, 365, by=30))+
  labs(title= "Dipteryx oleifera: Leaf coverage by tree and year",
       subtitle= "each color is a tree, each panel is a year")+
  ylab("Leaf coverage (normalized)")+
  xlab("Days since Oct 1st")




doubleLogisticFixedRandom<- brm(
    form_main,
    data = data,
    prior = priors,
    family = Beta(),
    control = list(adapt_delta = 0.95, max_treedepth = 15),
    iter= 4000,
    warmup= 2000,
    chains= 4,
    cores= 4
)

?brm

rm(post)

post <- as_draws_df(doubleLogisticFixedRandom)
colnames_post<-colnames(post)
#timing parameter td
td_year<- post[,grepl("b_td", colnames_post)]    #there is 7 years here
td_tree<- post[,grepl("r_tree__td", colnames_post)] #there is 15 trees here
td_intercept<- post[,grepl("b_td_Intercept", colnames_post)]

delta_year<- post[,grepl("b_delta", colnames_post)] #there is 7 years here
delta_tree<- post[,grepl("r_tree__delta", colnames_post)] #there is 15 trees here
delta_intercept<- post[,grepl("b_delta_Intercept", colnames_post)]

## change the col names to be more readable
colnames(td_tree)<- gsub("r_tree__td\\[(.*),.*", "\\1", colnames(td_tree))
colnames(td_year)<- gsub("b_td_pheno_year(\\d{4})", "\\1", colnames(td_year))
colnames(td_year)[1] <- "2017"

colnames(delta_tree)<- gsub("r_tree__delta\\[(.*),.*", "\\1", colnames(delta_tree))
colnames(delta_year)<- gsub("b_delta_pheno_year(\\d{4})", "\\1", colnames(delta_year))
colnames(delta_year)[1] <- "2017"

year_names <- colnames(td_year)
tree_names <- colnames(td_tree)

##construct the tf
tf_intercept<- td_intercept + delta_intercept   #you have to construct tf before adding the intercept to td and delta
tf_tree<- td_tree + delta_tree
tf_year<- td_year + delta_year
#################################################

delta_combinations<- data.frame(rows= 1:nrow(delta_year))
for (i in year_names){
  for (j in tree_names){
    delta_combinations[,paste(j,i, sep="_")] <- delta_intercept$b_delta_Intercept +delta_year[,i] + delta_tree[,j]## here there delta doesnt needs the intercept
  }
}

td_combinations<- data.frame(rows= 1:nrow(td_year))
for (i in year_names){
  for (j in tree_names){
    td_combinations[,paste(j,i, sep="_")] <- td_intercept$b_td_Intercept + td_year[,i] + td_tree[,j]  ## intercept is td_year[,1]
  }
}

tf_combinations<- data.frame(rows= 1:nrow(tf_year))
for (i in year_names){
  for (j in tree_names){
    tf_combinations[,paste(j,i, sep="_")] <- td_intercept$b_td_Intercept + td_year[,i] + td_tree[,j] + delta_year[,i] + delta_tree[,j] ## intercept is td_year[,1]
  }
}

delta_combinations<- delta_combinations[,-1]
td_combinations<- td_combinations[,-1]
tf_combinations<- tf_combinations[,-1]

td_long<- td_combinations %>%
  pivot_longer(
    cols = everything(),
    names_to = "year_tree",
    values_to = "value"
  ) %>% separate(year_tree, into = c("tree", "pheno_year"), sep = "_", remove = FALSE)%>%
  mutate(type= "td")

tf_long<- tf_combinations %>%
  pivot_longer(
    cols = everything(),
    names_to = "year_tree",
    values_to = "value"
  ) %>% separate(year_tree, into = c("tree", "pheno_year"), sep = "_", remove = FALSE)%>%
  mutate(type= "tf")

delta_long<- delta_combinations %>%
  pivot_longer(
    cols = everything(),
    names_to = "year_tree",
    values_to = "value"
  ) %>% separate(year_tree, into = c("tree", "pheno_year"), sep = "_", remove = FALSE)%>%
  mutate(type= "delta")

td_tf<- rbind(td_long, tf_long)
windows()
ggplot(delta_long, aes(x=value, fill=tree))+
  geom_histogram(aes(y= (..density..)), bins=100)+
  geom_line(data=data, aes(x=day, y=y_norm, color=tree), alpha=0.5, inherit.aes = FALSE, show.legend= FALSE)+
  facet_wrap(~pheno_year, ncol=1)







for (i in 1:ncol(td_tree)){
    td_tree[,i]<- td_tree[,i] + td_intercept  # add the intercept to each tree
}




tf_intercept<- td_intercept + delta_intercept   #you have to construct tf before adding the intercept to td and delta
tf_tree<- td_tree + delta_tree
tf_year<- td_year + delta_year

)
##################################################


for (i in 1:ncol(tf_tree)){
    tf_tree[,i]<- tf_tree[,i] + tf_intercept  # add the intercept to each tree
}

# for (i in 1:ncol(delta_tree)){
#     delta_tree[,i]<- delta_tree[,i] + delta_intercept  # add the intercept to each tree
# }

#calculate the posterior for each combination of tree and year


delta_combinations<- data.frame(rows= 1:nrow(delta_year))
for (i in 1:ncol(delta_tree)){
  for (j in 1:ncol(delta_year)){
    delta_combinations[,paste(tree_names[i], year_names[j], sep="_")] <- delta_year[,j] + delta_tree[,i]## intercept is delta_year[,1]
  }
}

delta_combinations<- delta_combinations[,-1]
tf_combinations<- tf_combinations[,-1]
td_combinations<- td_combinations[,-1]

td_long<- td_combinations %>%
  pivot_longer(
    cols = everything(),
    names_to = "tree_year",
    values_to = "value"
  ) %>% separate(tree_year, into = c("tree", "pheno_year"), sep = "_", remove = FALSE)%>%
  mutate(type= "td")

tf_long<- tf_combinations %>%
  pivot_longer(
    cols = everything(),
    names_to = "tree_year",
    values_to = "value"
  ) %>% separate(tree_year, into = c("tree", "pheno_year"), sep = "_", remove = FALSE)%>%
  mutate(type= "tf")

delta_long<- delta_combinations %>%
  pivot_longer(
    cols = everything(),
    names_to = "tree_year",
    values_to = "value"
  ) %>% separate(tree_year, into = c("tree", "pheno_year"), sep = "_", remove = FALSE)%>%
  mutate(type= "delta")

windows()
ggplot(td_long, aes(x=value, fill=tree))+
  geom_histogram(aes(y= (..density..)), bins=100)+
  facet_wrap(~pheno_year, ncol=1)

windows()
ggplot(tf_long, aes(x=value, fill=tree))+
  geom_histogram(aes(y= (..density..)), bins=100)+
  facet_wrap(~pheno_year, ncol=1)


windows()
ggplot(delta_long, aes(x= value, y= tree, fill= tree))+
  geom_boxplot(alpha= 0.7, outlier.size = 0.5)+
  facet_wrap(~pheno_year, ncol=1)+
  scale_fill_viridis_d(name = "Tree tag")+
  theme_minimal()+
  labs(title= "Cavallinesia planatifolia: Duration by tree and year")+
  xlab("Duration (days below 50%)")+
  ylab("Tree tag")













td_tf<- rbind(td_long, tf_long)


# for delta combinations i want for every year three segments. one for the median delta, another for the 5% and another for the 95%
delta_summary<- delta_long %>%
    group_by(pheno_year) %>%
    summarise(
        median= median(value),
        p5= quantile(value, 0.05),
        p95= quantile(value, 0.95)
    )

td_summary <- td_long %>%
  group_by(pheno_year, tree) %>%
  summarise(
    peak_density_y = {
      d <- density(value)
      max(d$y)  # maximum density value for each tree
    },
    p5 = quantile(value, 0.05),
    median = quantile(value, 0.5),
    p95 = quantile(value, 0.95),
    .groups = "drop_last"  # Keep pheno_year grouping
  ) %>%
  group_by(pheno_year) %>%
  summarise(
    sum_peak_density = sum(peak_density_y),  # Sum all trees' peak densities per year
    p5_mean= mean(p5),
    median_mean= mean(median),
    p95_mean= mean(p95)
  )


segments <- td_summary %>%
  left_join(delta_summary, by = c("pheno_year")) %>%
  mutate(
    x_start_median = median_mean,
    x_end_median = median_mean + median,
    x_start_p5 = p5_mean,
    x_end_p5 = p5_mean + p5,
    x_start_p95 = p95_mean,
    x_end_p95 = p95_mean + p95
  )%>%
  pivot_longer(
    cols = starts_with("x_"),
    names_to = c(".value", "stat"),
    names_pattern = "x_(start|end)_(.*)"
  ) %>%
  select(pheno_year, stat, start, end,sum_peak_density) %>%
  mutate(
    ystart = case_when(
      stat == "median" ~ sum_peak_density,
      stat == "p5" ~ 0.1,
      stat == "p95" ~ 0.3,
      TRUE ~ NA_real_  # fallback for any other values
    ),
    yend = ystart
  )


start_date <- as.Date("2017-09-01") # the origin of the transformed day variable
breaks <- as.numeric(difftime(
  seq(start_date, by = "1 months", length.out = 12),
  start_date,
  units = "days"
))
labels <- format(start_date + breaks, "%b")  


windows()
ggplot(td_tf, aes(x=value, fill=tree, alpha=type))+
  geom_histogram(aes(y= (..density..)), bins=100)+
  geom_segment(data=segments, aes(x=start, y=ystart, xend=end, yend=yend, color=stat), size=1, inherit.aes = FALSE)+
  geom_line(data=data[data$tree != 134166,], aes(x=day, y=y_norm, color=tree), alpha=0.5, inherit.aes = FALSE, show.legend= FALSE)+
  facet_wrap(~pheno_year, ncol=1)+
  scale_alpha_manual(values = c("td" = 0.7, "tf" = 0.4), 
                     labels = c("td" = "Leaf drop", "tf" = "Leaf flush"),
                     name = "Type")+
  scale_color_manual(values = c("p5" = "red", "median" = "black", "p95" = "blue"), 
                     name = "Deciduousness duration")+
  scale_fill_viridis_d(name = "Tree tag")+
  theme_minimal()+
  labs(title= "Cavallinesia planatifolia: timing and duration")+
  ylab("Density and Leaf Coverage")+
  xlab("Days")+
  scale_x_continuous(
    breaks = breaks,
    labels = labels
  )

windows()
ggplot(delta_long, aes(x=value, fill=tree))+
  geom_histogram(aes(y= (..density..)), bins=100)

#lets make a delta long horizontal boxplot with facets per year

windows()
ggplot(td_long, aes(x= value, y= tree, fill= tree))+
  geom_boxplot(alpha= 0.7, outlier.size = 0.5)+
  facet_wrap(~pheno_year, ncol=1)+
  scale_fill_viridis_d(name = "Tree tag")+
  theme_minimal()+
  labs(title= "Cavallinesia planatifolia: Leaf drop timing by tree and year")+
  xlab("Leaf drop timing (days since Sept 1)")+
  ylab("Tree tag")+
  scale_x_continuous(
    breaks = breaks,
    labels = labels
  )

windows()
ggplot(tf_long, aes(x= value, y= tree, fill= tree))+
  geom_boxplot(alpha= 0.7, outlier.size = 0.5)+
  facet_wrap(~pheno_year, ncol=1)+
  scale_fill_viridis_d(name = "Tree tag")+
  theme_minimal()+
  labs(title= "Cavallinesia planatifolia: Leaf flush timing by tree and year")+
  xlab("Leaf flush timing (days since Sept 1)")+
  ylab("Tree tag")+
  scale_x_continuous(
    breaks = breaks,
    labels = labels
  )
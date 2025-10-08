library(pacman)
p_load(dplyr, lubridate, ggplot2,brms,tidyr, modeest)

f_decay <- function(values, k_d, t_d){
    1/ (1 + exp(k_d *(values - t_d)))
}

f_flush <- function(values, k_f, t_f){
    1/ (1 + exp(-k_f * (values - t_f)))
}

f_phase <- function(values, tm){
    1/ (1 + exp(-0.5 * (values - tm)))   
}

values <- seq(0, 70, by=1)
decay_values <- f_decay(values, k_d=0.8, t_d=20)
flush_values <- f_flush(values, k_f=0.8, t_f=50)


LC<- function(values, k_d, t_d, k_f, t_f){
    t_m<- (t_d + t_f) / 2
    values<- (1-f_phase(values, t_m)) * f_decay(values, k_d, t_d) + f_phase(values, t_m) * f_flush(values, k_f, t_f)
    return(values)
}

LC_values<- LC(values, k_d=0.8, t_d=20, k_f=0.8, t_f=50)

windows()
plot(values, LC_values, type='l', col='black', ylim=c(0,1))



form_double_reparam <- bf(
  y_norm ~ (1 - (1 / (1 + exp(-0.5 * (day - (td + delta/2)))))) *
             (1 / (1 + exp(kd * (day - td)))) +
           (1 / (1 + exp(-0.5 * (day - (td + delta/2)))) ) *
             (1 / (1 + exp(-kf * (day - (td + delta))))),
  kd + kf + td + delta ~ 1 + (1 | pheno_year) + (1 | tree),
  nl = TRUE
)

form_double_inter <- bf(
  y_norm ~ (1 - (1 / (1 + exp(-0.5 * (day - (td + delta/2)))))) *
             (1 / (1 + exp(kd * (day - td)))) +
           (1 / (1 + exp(-0.5 * (day - (td + delta/2)))) ) *
             (1 / (1 + exp(-kf * (day - (td + delta))))),
  kd + kf + td + delta ~ 1 + (1 | tree_year),
  nl = TRUE
)

form_double_fixedRandom <- bf(
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


windows()
ggplot(data[data$tree== 134166,], aes(x=day, y=y_norm, color= tree))+
  geom_line()+
  facet_wrap(~pheno_year)+
  scale_x_continuous(breaks= seq(0, 365, by=5))+
  theme(legend.position = "none")

View(data)
priors <- c(
  prior(normal(130, 6), nlpar = "td", lb = 0),
  prior(normal(120, 15), nlpar = "delta", lb = 0),  # tf ~ 130 + 120 = 250
  prior(normal(0.5, 0.25), nlpar = "kd", lb = 0),
  prior(normal(0.5, 0.25), nlpar = "kf", lb = 0),

  # group sd priors to avoid huge offsets
  prior(exponential(0.4), class = "sd", nlpar = "td"),
  prior(exponential(0.4), class = "sd", nlpar = "delta")
)

rm(doubleLogisticPhased)

doubleLogisticPhased<- brm(
    form_double_reparam,
    data = data,
    family = Beta(),
    prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 15),
    iter= 4000,
    warmup= 2000,
    chains= 4,
    cores= 4
)

doubleLogisticFixedRandom<- brm(
    form_double_fixedRandom,
    data = data,
    family = Beta(),
    prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 15),
    iter= 4000,
    warmup= 2000,
    chains= 4,
    cores= 4
)

rm(post)

post <- as_draws_df(doubleLogisticFixedRandom)
colnames_post<-colnames(post)

td_year<- post[,grepl("b_td_pheno_year", colnames_post)]
td_tree<- post[,grepl("r_tree__td", colnames_post)]

delta_year<- post[,grepl("r_pheno_year__delta", colnames_post)]
delta_tree<- post[,grepl("r_tree__delta", colnames_post)]

td_treeYear<- post[,grepl("r_tree_year__td", colnames_post)]
td_intercept<- post[,"b_td_Intercept"]
delta_intercept<- post[,"b_delta_Intercept"]



for (i in 1:ncol(td_tree)){
  td_tree[,i]<- td_tree[,i] + td_year$b_td_pheno_year2021
}

for (i in 1:ncol(td_year)){
  #if td_year[,i] is not the intercept column
  if (colnames(td_year)[i] == "b_td_Intercept") next
  td_year[,i]<- td_year[,i] + td_intercept
}

for (i in 1:ncol(delta_tree)){
  #if delta_tree[,i] is not the intercept column
  if (colnames(delta_tree)[i] == "b_delta_Intercept") next
  delta_tree[,i]<- delta_tree[,i] + delta_intercept
}
for (i in 1:ncol(delta_year)){
  #if delta_year[,i] is not the intercept column
  if (colnames(delta_year)[i] == "b_delta_Intercept") next
  delta_year[,i]<- delta_year[,i] + delta_intercept
}

treeYear_long<- td_treeYear %>%
  pivot_longer(
    cols = everything(),
    names_to = "tree_year",
    values_to = "td"
  ) %>%
  mutate(tree_year = gsub("r_tree_year__td\\[(.*),.*", "\\1", tree_year))

treeYear_long<- treeYear_long %>%
  separate(tree_year, into = c("tree", "year"), sep = "_")%>%
  mutate(treeYear= paste(tree, year, sep="_"))

windows()
ggplot(treeYear_long, aes(x=td, fill=treeYear))+
  geom_histogram(aes(y= (..density..)), bins=100, alpha=0.7)+
  theme(legend.position = "none")

delta_year_long<- delta_year %>%
  pivot_longer(
    cols = everything(),
    names_to = "year",
    values_to = "delta"
  ) %>%
  mutate(year = gsub("r_pheno_year__delta\\[(\\d{4}),.*", "\\1", year))

td_year_long<- td_year %>%
  pivot_longer(
    cols = everything(),
    names_to = "pheno_year",
    values_to = "td"
  ) %>%
  mutate(pheno_year = gsub("b_td_pheno_year(\\d{4})", "\\1", pheno_year))

rm(td_tree_long)
td_tree_long<- td_tree %>%
  pivot_longer(
    cols = everything(),
    names_to = "tree",
    values_to = "td"
  ) %>%
  mutate(tree = gsub("r_tree__td\\[(\\d{3,6}),.*", "\\1", tree)) %>%
  filter(tree != 134166)

windows()
ggplot(td_tree_long, aes(x=td, fill=tree))+
  geom_histogram(aes(y= (..density..)*10), bins=100, alpha=0.7)+
  geom_line(data=data, aes(x=day, y=y_norm, color=tree), alpha=0.5, inherit.aes = FALSE)+
  facet_wrap(~tree)

windows()
ggplot(data[data$pheno_year==2021,], aes(x=day, y=y_norm, color=tree))+
  geom_line()+
  geom_histogram(data= td_tree_long, aes(x=td, y=(..density..)*10, color=tree), bins=100, alpha=0.5, inherit.aes = FALSE)+
  facet_wrap(~tree)

summary_td <- td_year_long %>%
  group_by(pheno_year) %>%
  summarise(
    mean_td = mean(td),
    median_td = median(td),
    lower_95 = quantile(td, 0.025),
    upper_95 = quantile(td, 0.975),
    sd_td = sd(td),
    mode_td = mlv(td, method = "shorth")$M
  )


windows()
ggplot(td_year_long, aes(x = td, fill = pheno_year)) +
  geom_histogram(aes(y = (..density..)*10), bins = 100, alpha = 0.7) +
  geom_line(data=data, aes(x = day, y = y_norm, color = tree), alpha = 0.5, inherit.aes = FALSE) +
  facet_wrap(~pheno_year) +
  ylab("Normalized count")


windows()
ggplot(td_year_long, aes(x=td, fill=year))+
  geom_histogram(bins=100)+
  facet_wrap(~year, ncol=1, scales = "fixed")

windows()
ggplot(delta_year_long, aes(x=delta, fill=year))+
  geom_histogram(bins=100)+
  facet_wrap(~year, ncol=1, scales = "fixed")





td_tree_long<- td_tree %>%
  pivot_longer(
    cols = everything(),
    names_to = "tree",
    values_to = "td"
  ) %>%
  mutate(tree = gsub("r_tree__td\\[([A-Z]),.*", "\\1", tree))

windows()
ggplot(td_tree_long, aes(x=td, fill=tree))+
  geom_histogram(bins=200)+
  facet_wrap(~tree, ncol=1, scales = "fixed")+
  theme(strip.text = element_blank())+
  xlim(-5,5)


td_year<- post[,grepl("b_td", colnames_post)]
#drop the intercept column
td_year_intercept<- post[,"b_td_Intercept"]



for (i in 1:ncol(td_year)){
    #if td_year[,i] is not the intercept column
    if (colnames(td_year)[i] == "b_td_Intercept") next
    td_year[,i]<- td_year[,i] + td_year_intercept
}


tf_year<- post[,grepl("b_tf", colnames_post)]
tf_year_intercept<- post[,"b_tf_Intercept"]

for (i in 1:ncol(tf_year)){
  tf_year[,i]<- tf_year[,i] + tf_year_intercept
}


td_year_long<- td_year %>%
  pivot_longer(
    cols = everything(),
    names_to = "year",
    values_to = "td"
  ) %>%
  mutate(year = gsub("b_td_", "", year)) 

tf_year_long<- tf_year %>%
  pivot_longer(
    cols = everything(),
    names_to = "year",
    values_to = "tf"
  ) %>%
  mutate(year = gsub("b_tf_", "", year))


windows()
ggplot(td_year_long, aes(x=td, fill=year))+
  geom_histogram(bins=200)+
  facet_wrap(~year, ncol=1, scales = "fixed")

# Select the intercept and year effects
td_post <- post %>%
  select(
    "b_td_Intercept",
    "r_year__td[2018,Intercept]",
    "r_year__td[2019,Intercept]",
    "r_year__td[2020,Intercept]",
    "r_year__td[2021,Intercept]",
    "r_year__td[2022,Intercept]",
    "r_year__td[2023,Intercept]",
    "r_year__td[2024,Intercept]"
  )

td_post_tree <- post %>%
  select(
    "b_td_Intercept",
    "r_tree__td[A,Intercept]",
    "r_tree__td[B,Intercept]",
    "r_tree__td[C,Intercept]",
    "r_tree__td[D,Intercept]",
    "r_tree__td[E,Intercept]",
    "r_tree__td[F,Intercept]",
    "r_tree__td[G,Intercept]",
    "r_tree__td[H,Intercept]",
    "r_tree__td[I,Intercept]",
    "r_tree__td[J,Intercept]",
    "r_tree__td[K,Intercept]",
    "r_tree__td[L,Intercept]",
    "r_tree__td[M,Intercept]",
    "r_tree__td[N,Intercept]",
    "r_tree__td[O,Intercept]",
    "r_tree__td[P,Intercept]",
    "r_tree__td[Q,Intercept]",
    "r_tree__td[R,Intercept]",
    "r_tree__td[S,Intercept]",
    "r_tree__td[T,Intercept]"
  )

# Pivot to long format
td_calc <- td_post %>%
  mutate(
    td_2018 = b_td_Intercept + `r_year__td[2018,Intercept]`,
    td_2019 = b_td_Intercept + `r_year__td[2019,Intercept]`,
    td_2020 = b_td_Intercept + `r_year__td[2020,Intercept]`,
    td_2021 = b_td_Intercept + `r_year__td[2021,Intercept]`,
    td_2022 = b_td_Intercept + `r_year__td[2022,Intercept]`,
    td_2023 = b_td_Intercept + `r_year__td[2023,Intercept]`,
    td_2024 = b_td_Intercept + `r_year__td[2024,Intercept]`
  ) %>%
  select(starts_with("td_"))

td_calc_tree <- td_post_tree %>%
  mutate(
    td_A = b_td_Intercept + `r_tree__td[A,Intercept]`,
    td_B = b_td_Intercept + `r_tree__td[B,Intercept]`,
    td_C = b_td_Intercept + `r_tree__td[C,Intercept]`,
    td_D = b_td_Intercept + `r_tree__td[D,Intercept]`,
    td_E = b_td_Intercept + `r_tree__td[E,Intercept]`,
    td_F = b_td_Intercept + `r_tree__td[F,Intercept]`,
    td_G = b_td_Intercept + `r_tree__td[G,Intercept]`,
    td_H = b_td_Intercept + `r_tree__td[H,Intercept]`,
    td_I = b_td_Intercept + `r_tree__td[I,Intercept]`,
    td_J = b_td_Intercept + `r_tree__td[J,Intercept]`,
    td_K = b_td_Intercept + `r_tree__td[K,Intercept]`,
    td_L = b_td_Intercept + `r_tree__td[L,Intercept]`,
    td_M = b_td_Intercept + `r_tree__td[M,Intercept]`,
    td_N = b_td_Intercept + `r_tree__td[N,Intercept]`,
    td_O = b_td_Intercept + `r_tree__td[O,Intercept]`,
    td_P = b_td_Intercept + `r_tree__td[P,Intercept]`,
    td_Q = b_td_Intercept + `r_tree__td[Q,Intercept]`,
    td_R = b_td_Intercept + `r_tree__td[R,Intercept]`,
    td_S = b_td_Intercept + `r_tree__td[S,Intercept]`,
    td_T = b_td_Intercept + `r_tree__td[T,Intercept]`
  ) %>%
  select(starts_with("td_"))

# Pivot to long format
td_long <- td_calc %>%
  pivot_longer(
    cols = everything(),
    names_to = "year",
    values_to = "td"
  ) %>%
  mutate(year = gsub("td_", "", year))

td_long_tree <- td_calc_tree %>%
  pivot_longer(
    cols = everything(),
    names_to = "tree",
    values_to = "td"
  ) %>%
  mutate(tree = gsub("td_", "", tree))

# Plot histograms
windows()
ggplot(td_long, aes(x = td, fill = year)) +
  geom_histogram(bins = 100) +
  facet_wrap(~ year, ncol = 1, scales = "fixed")+
  xlim(0,50)

windows()
ggplot(td_long_tree, aes(x = td, fill = tree)) +
  geom_histogram(bins = 100) +
  facet_wrap(~ tree, ncol = 1, scales = "fixed")+
  xlim(0,50)+
  theme(strip.text = element_blank())
#how late is td 2021 compared to all other years?
# Test if 2021 is later than 2020

############this is comparing years#########
##summarize each of the years in td_long
summary_td <- td_long %>%
  group_by(year) %>%
  summarize(
    mean_td = mean(td),
    median_td = median(td),
    sd_td = sd(td),
    n = n(),
    fifthPercentile = quantile(td, 0.05),
    ninetyFifthPercentile = quantile(td, 0.95)
  )
View(summary_td)

windows()
ggplot(td_long, aes(x = td, fill = year)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 100) +
  theme_minimal() +
  labs(x = "td posterior", y = "Frequency", fill = "Year") +
  scale_fill_brewer(palette = "Set1")+
  xlim(0,50


############end#################

colnames_post<- colnames(post)

post_td_2021<- post_td[,grep("2021", colnames(post_td))]
colnames(post_td_2021)
post<- as_draws_df(doubleLogisticPhased)
colnames_post<- colnames(post)
post_td<- post[,grep("td", colnames_post)]
post_td_2021_long<- as.data.frame(post_td) %>%
  pivot_longer(
    cols= everything(),
    names_to= "td_name",
    values_to= "td"
  ) %>% 
  mutate(tree= gsub("r_treeYear__td\\[([A-Z])_.*", "\\1", td_name),
         year= gsub("r_treeYear__td\\[[A-Z]_(\\d{4}),.*", "\\1", td_name))%>%
    filter(year %in% c(2018,2019,2020,2021,2022,2023,2024))


windows()
ggplot(post_td_2021_long, aes(x=td, fill=tree))+
  geom_histogram(bins=200)+
  facet_wrap(~year, ncol=1, scales = "fixed")+
    xlim(-5,5)+
    theme(strip.text = element_blank())
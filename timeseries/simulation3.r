library(pacman)
p_load(dplyr, lubridate, ggplot2,brms,tidyr)

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

windows()
plot(values, decay_values, type='l', col='blue', ylim=c(0,1))
lines(values, flush_values, col='red')

LC<- function(values, k_d, t_d, k_f, t_f){
    t_m<- (t_d + t_f) / 2
    values<- (1-f_phase(values, t_m)) * f_decay(values, k_d, t_d) + f_phase(values, t_m) * f_flush(values, k_f, t_f)
    return(values)
}

LC_values<- LC(values, k_d=0.8, t_d=20, k_f=0.8, t_f=50)

windows()
plot(values, LC_values, type='l', col='black', ylim=c(0,1))



form <- bf(
    y_norm ~ (1 - ( 1 / (1 + exp(-0.5 * (day - (td + tf)/2))))) * ( 1 / (1 + exp(kd * (day - td)))) + (1 / (1 + exp(-0.5 * (day - (td + tf)/2)))) * (1 / (1 + exp(-kf * (day - tf)))),
    kd + kf + td + tf ~ 1 + (1 | year) + (1 | tree),
    nl = TRUE
)


simulated<- read.csv("timeseries\\simulated_phenophase_data.csv")
simulated<- simulated %>%
    mutate(day= yday(date),
             year= year(date),
             y_norm = pmin(pmax(observed_leafing / 100, 1e-4), 1 - 1e-4),
             treeYear= paste(tree, year, sep="_"))
windows()
ggplot(simulated[simulated$tree %in% c("A", "E"),], aes(x=day, y=y_norm, shap=treeYear))+
    geom_line()+
    theme_minimal()+
    xlim(0,100)

priors <- c(
  prior(normal(25, 3), nlpar = "td"),  # ~95% in [10, 40]
  prior(normal(50, 3), nlpar = "tf"),    # ~95% in [30, 70]
  prior(normal(0.8, 0.2), nlpar = "kd"),
  prior(normal(0.8, 0.2), nlpar = "kf")
)

doubleLogisticPhased<- brm(
    form,
    data = simulated,
    family = Beta(),
    prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 15),
    iter= 4000,
    warmup= 2000,
    chains= 4,
    cores= 4
)

post <- as_draws_df(doubleLogisticPhased)
colnames(post)

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

td_E_2021 <- post[ , "b_td_Intercept"] +
            post[ , "r_tree__td[E,Intercept]"] +
            post[ , "r_year__td[2021,Intercept]"]

td_A_2020 <- post[ , "b_td_Intercept"] +
            post[ , "r_tree__td[A,Intercept]"] +
            post[ , "r_year__td[2020,Intercept]"]

df<- data.frame(b_td_Intercept= post[ , "b_td_Intercept"],
                td_E_2021= td_E_2021,
                td_A_2020= td_A_2020)
head(df)
windows()
hist(td_A_2020[1:8000,],breaks=100,xlim=c(0,50))
hist(td_E_2021[1:8000,],breaks=100,col=rgb(1,0,0,0.5),add=T)


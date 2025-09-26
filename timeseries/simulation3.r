library(pacman)
p_load(dplyr, lubridate, ggplot2,brms)

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
    kd + kf + td + tf ~ 1 + (1 | year),
    nl = TRUE
)


simulated<- read.csv("timeseries\\simulated_phenophase_data.csv")
simulated<- simulated %>%
    mutate(day= yday(date),
             year= year(date),
             y_norm = pmin(pmax(observed_leafing / 100, 1e-4), 1 - 1e-4))


priors <- c(
  prior(normal(25, 2), nlpar = "td"),  # ~95% in [10, 40]
  prior(normal(50, 2), nlpar = "tf"),    # ~95% in [30, 70]
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

# Pivot to long format
td_long <- td_calc %>%
  pivot_longer(
    cols = everything(),
    names_to = "year",
    values_to = "td"
  ) %>%
  mutate(year = gsub("td_", "", year))

# Plot histograms
windows()
ggplot(td_long, aes(x = td, fill = year)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 100) +
  theme_minimal() +
  labs(x = "td posterior", y = "Frequency", fill = "Year") +
  scale_fill_brewer(palette = "Set1")+
  xlim(0,50)


#how late is td 2021 compared to all other years?
# Test if 2021 is later than 2020

post <- as_draws_df(doubleLogisticPhased)

td_2021 <- post$b_td_Intercept + post$`r_year__td[2021,Intercept]`
td_2020 <- post$b_td_Intercept + post$`r_year__td[2020,Intercept]`

diff_2021_2020 <- td_2021 - td_2020

# posterior probability that 2021 is later than 2020
mean(diff_2021_2020 > 0)

mean(diff_2021_2020 )

# summary
quantile(diff_2021_2020, c(0.025, 0.5, 0.975))

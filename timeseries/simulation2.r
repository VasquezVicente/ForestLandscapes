library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan, lme4, circular, bpnreg,tidyr, tidyverse, brms)

# Convert to DOY
radians_to_days <- function(r) {
  (r / (2 * pi)) * 365
}

days_to_radians <- function(d) {
  (d / 365) * 2 * pi
}

simulated<- read.csv("timeseries\\simulated_phenophase_data.csv")
simulated$day<- yday(simulated$date)
simulated$year<- year(simulated$date)
simulated$month<- month(simulated$date)
simulated$is_leap<- leap_year(simulated$year)

simulated$oct1 <- ifelse(simulated$is_leap, 275, 274)
simulated$phenoDay <- simulated$day - simulated$oct1 + 1
simulated$phenoDay <- ifelse(simulated$phenoDay <= 0,
                             simulated$phenoDay + ifelse(simulated$is_leap, 366, 365),
                             simulated$phenoDay)

View(simulated)
head(simulated)

windows()
ggplot(simulated, aes(x=phenoDay, y=observed_leafing, color=as.factor(year)))+
  geom_point()+
  scale_x_continuous(name="Pheno Day",
                     breaks=c(1,20,40,60,70,80,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360))+
  theme_minimal()

ggplot(simulated, aes(x = phenoDay, y = observed_leafing, color = as.factor(year))) +
  geom_point() +
  scale_x_continuous(
    name = "DOY",
    breaks = c(1, 60, 120, 180, 240, 300, 365),
    labels = function(x) {
      doy <- (x + simulated$oct1[1] - 1) %% ifelse(simulated$is_leap[1], 366, 365)
      ifelse(doy == 0, ifelse(simulated$is_leap[1], 366, 365), doy)
    }
  ) +
  theme_minimal()

fit_leafing<- bf(observed_leafing ~ Lmax / (1 + exp(-k * (phenoDay - t0))),
                 Lmax ~ 1,
                 k ~ 1,
                 t0 ~ 1 + (1|year),
                 nl = TRUE)

priors <- c(
  prior(normal(100, 5), nlpar = "Lmax"),    # near 100
  prior(normal(0.88, 0.3), nlpar = "k"),    # slope
  prior(normal(120, 20), nlpar = "t0"))      # midpoint prior around true value


fit_brms <- brm(
  formula = fit_leafing,
  data = simulated,
  prior = priors,
  chains = 4,
  iter = 4000,
  cores = 4,
  control = list(adapt_delta = 0.95)
)

posterior<- as_draws_df(fit_brms)
colnames(posterior)

posterior_2019<- as_draws_df(fit_brms, variable= "b_t0_Intercept")
summary(posterior_2019)

fit_brms.
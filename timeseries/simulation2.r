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
simulated$dip <- pmax(0, 100 - simulated$observed_leafing)
head(simulated)

windows()
ggplot(simulated, aes(x=phenoDay, y=dip, color=as.factor(year)))+
  geom_point()+
  scale_x_continuous(name="Pheno Day",
                     breaks=c(1,20,40,60,70,80,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360))+
  theme_minimal()


simulated$y_norm <- pmin(pmax(simulated$observed_leafing / 100, 1e-4), 1 - 1e-4)

head(simulated)

fit <- brm(
  bf(y_norm ~ s(day, bs = "cc", k = 20) + (1 | year)),
  data = simulated,
  family = Beta(),
  chains = 4, iter = 4000, warmup = 1000
)


# Predict the latent curve across days (population mean, no random effects)
newdays <- data.frame(day = 1:365)
mu_pop <- posterior_epred(fit, newdata = newdays, re_formula = NA)


# Convert back to 0-100
latent_curve <- 100 * apply(mu_pop, 2, mean)        # mean curve
lat_lower   <- 100 * apply(mu_pop, 2, quantile, probs = 0.025)
lat_upper   <- 100 * apply(mu_pop, 2, quantile, probs = 0.975)

# Plot (example)
windows()
plot(simulated$day, simulated$observed_leafing, pch = 16, cex = 0.5, col = rgb(0,0,0,0.1), # nolint
     xlab = "Day of year", ylab = "Predicted leafing (0-100)")
lines(1:365, latent_curve, lwd = 2, col = "darkgreen")
lines(1:365, lat_lower, lty = 2, col = "gray50")
lines(1:365, lat_upper, lty = 2, col = "gray50")

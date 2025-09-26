library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan, lme4, circular, bpnreg,tidyr, tidyverse, brms)

y <- c(rep(100, 20), 95, 85, 65, 40, 20, 10, 0, rep(0, 20), 5, 25, 45, 70, 90, 100, rep(100, 313))
t <- 1:366
df_true <- data.frame(time = t, leafing = y)

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
simulated$y_norm <- pmin(pmax(simulated$observed_leafing / 100, 1e-4), 1 - 1e-4)


simulated$is_leap<- leap_year(simulated$year)

simulated$oct1 <- ifelse(simulated$is_leap, 275, 274)
simulated$phenoDay <- simulated$day - simulated$oct1 + 1
simulated$phenoDay <- ifelse(simulated$phenoDay <= 0,
                             simulated$phenoDay + ifelse(simulated$is_leap, 366, 365),
                             simulated$phenoDay)

windows()
ggplot(simulated, aes(x=phenoDay, y=observed_leafing, color=as.factor(year)))+
  geom_point()+
  scale_x_continuous(name="Pheno Day",
                     breaks=c(1,20,40,60,70,80,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360))+
  theme_minimal()


head(simulated)

# the cyclic spline model succesfully models the data but does not provide parameters of interest
# such as the timing of leaf drop and leaf out
# it also does not capture variation accross years since is ciclic and smooth we cann
fit_smooth <- brm(
  y_norm ~ s(day,k=20, bs="cc"),
  data = simulated,
  family = Beta(),
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4
)

newdata <- data.frame(day = 1:365)
pred_draws <- posterior_epred(fit_smooth, newdata = newdata)
pred_mean <- posterior_epred(fit_smooth, newdata = newdata, re_formula= NA)
pred_mean <- colMeans(pred_mean)
first_below_09 <- which(pred_mean < 0.9)[1]
first_above_09 <- which(pred_mean > 0.9 & (1:365) > first_below_09)[1]

windows()
ggplot(newdata, aes(x = day, y = pred_mean)) +
  geom_line(color = "blue", size = 1) +
  geom_point(data = simulated, aes(x = day, y = y_norm, color = as.factor(year)), alpha = 0.5) +
  geom_line(data = df_true, aes(x = time, y = leafing / 100), color = "red", size = 1, linetype = "dashed") +
  geom_vline(xintercept = first_below_09, color = "black", size= 1) +
  geom_vline(xintercept = first_above_09,  color = "black", size= 1) +
  labs(x = "Day of Year", y = "Predicted y_norm") +
  theme_minimal() +
  xlim(1, 60)

newdata <- data.frame(day = 1:365)
pred_draws <- posterior_epred(fit_smooth, newdata = newdata, re_formula = NA) 
# pred_draws is draws x 365

# For each draw, compute first day below threshold (0.9):
threshold <- 0.9
first_below_per_draw <- apply(pred_draws, 1, function(row) {
  idx <- which(row < threshold)[1]
  if(is.na(idx)) return(NA) else return(idx)
})

# Summarize:
quantile(first_below_per_draw, probs = c(0.025, 0.5, 0.975), na.rm = TRUE)
table(first_below_per_draw) 



### one year is shifted 10 days we need to model it with a double sigmoid function

form <- bf(
  y_norm ~ (1 - (1 / (1 + exp(-kd * (day - td))))) + (1 / (1 + exp(-kf * (day - tf)))),
  kd + td + kf + tf ~ 1,
  nl = TRUE
)
priors <- c(
  prior(normal(1, 0.5), nlpar = "kd"),
  prior(normal(20, 10), nlpar = "td"),
  prior(normal(1, 0.5), nlpar = "kf"),
  prior(normal(50, 10), nlpar = "tf")
)
get_prior(form, data = simulated, family = Beta()) 
fit <- brm(
  form,
  data = simulated,
  family = Beta(),
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4
)

newdata <- data.frame(day = 1:365)
pred_draws <- posterior_epred(fit, newdata = newdata)
pred_mean <- posterior_epred(fit, newdata = newdata, re_formula = NA)
pred_mean <- colMeans(pred_mean) 

posterior <- as_draws_df(fit)
colnames(posterior)

windows()
hist(posterior$b_td_Intercept, breaks=30)
hist(posterior$b_tf_Intercept, breaks=30)
mean(posterior[posterior$b_td_Intercept<30,]$b_td_Intercept)
summary(fit)
windows()
ggplot(newdata, aes(x = day, y = pred_mean)) +
  geom_line(color = "blue", size = 1) +
  labs(x = "Day of Year", y = "Predicted y_norm") +
  theme_minimal()+
  xlim(10,40)








######################################################

##i need to pool all curves, one per year under the same grid


simulated<- read.csv("timeseries\\simulated_phenophase_data.csv")

simulated<- simulated %>%
  mutate(y_norm= pmin(pmax(observed_leafing / 100, 1e-4), 1 - 1e-4),
         day= yday(date),
         year= as.factor(year(date))

# we will fit a decay and a growth curve to each year

form<- bf(
  y_norm ~ (1 - (1 / (1 + exp(-kd * (day - td))))) + (1 / (1 + exp(-kf * (day - tf)))),
  kd + td + kf + tf ~ 1 + (1|year),
  nl = TRUE
)

priors <- c(
  prior(normal(1, 0.5), nlpar = "kd"),
  prior(normal(20, 10), nlpar = "td"),
  prior(normal(1, 0.5), nlpar = "kf"),
  prior(normal(50, 10), nlpar = "tf")
)

get_prior(form, data = simulated, family = Beta())

fit_double_logistic<- brm(
  form,
  data = simulated,
  family = Beta(),
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4
)
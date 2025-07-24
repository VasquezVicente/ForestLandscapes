library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan, lme4, circular, bpnreg,tidyr, tidyverse, brms)

# Convert to DOY
radians_to_days <- function(r) {
  (r / (2 * pi)) * 365
}

cavallinesia <- read.csv("timeseries/dataset_analysis/cavallinesia.csv")
cavallinesia$date<-as.Date(cavallinesia$date, format= "%Y-%m-%d")
cavallinesia$latin<- "Cavallinesia platanifolia"

##bpnreg radians~leaf coverage percentage

df <- cavallinesia %>% 
  mutate(
    year = as.integer(format(date, "%Y")),
    is_leap = (year %% 4 == 0 & year %% 100 != 0) | (year %% 400 == 0),
    dayOfYear  = yday(date),
    days_in_year = ifelse(is_leap, 366, 365),
    theta_rad = dayOfYear * 2 * pi / days_in_year,
    theta = ifelse(theta_rad > pi, 
                       theta_rad - 2 * pi, 
                       theta_rad),
    leafing= ifelse(leafing<0, 0, ifelse(leafing>100,100,leafing)),
    leafing_scaled= leafing/100
  )

View(df)
windows()
ggplot(data=df[df$GlobalID=='08c4ec87-3de9-40d7-b64e-d3a85534739e',], aes(x=date_num,y=leafing,color=GlobalID))+geom_line()


colnames(df)

fit_seasonal <- brm(
  leafing_scaled ~ s(dayOfYear, bs = "cc") + s(dayOfYear, GlobalID, bs = "fs") + (1 | year),
  data = df,
  family = zero_one_inflated_beta(),
  chains = 4,
  cores = 4,
  iter = 4000,
  control = list(adapt_delta = 0.99),
  prior = prior(normal(0, 2), class = "sds")
)


newdays <- data.frame(date_num = 1:365)
predicted <- predict(fit_leafing_gam, newdata = newdays, probs = c(0.025, 0.5, 0.975))

pred_df <- cbind(newdays, as.data.frame(predicted))


ggplot(pred_df, aes(x = date_num, y = Estimate)) +
  geom_line() +
  geom_ribbon(aes(ymin = Q2.5, ymax = Q97.5), alpha = 0.3) +
  facet_wrap(~ID, scales = "free_y") +
  labs(y = "Predicted Leafing", x = "Day", title = "Daily Leafing per Individual")


windows()
ggplot(data=df, aes(x=date,y=leafing, color=GlobalID))+geom_point()

View(df)
df$leafing<-as.numeric(df$leafing)
df$GlobalID_num<- as.numeric(as.factor(df$GlobalID))
df$GlobalID<- as.factor(df$GlobalID)

model <- bpnme(
  pred.I = theta_rad ~ leafing + (1 | GlobalID_num),
  data = df,
  its = 2000,
  burn = 200,
  n.lag = 5
)



windows()
traceplot(model, parameter="beta1")

radians_to_days(atan2(1.5784,0.5173))


leafing_vals <- 0:100
B01<- model$beta1[,"(Intercept)"]
B02<- model$beta2[,"(Intercept)"]

a1 <- model$beta1[,"leafing"]
a2 <- model$beta2[,"leafing"]
  

# Create an empty vector to store results
predicted_angles <- numeric()

# Loop over leafing values
for (x in leafing_vals) {
  theta_rads <- atan2(B02 + (a2 * x), B01 + (a1 * x))
  mean_angle <- mean(circular(theta_rads, modulo = "2pi"))
  theta = ifelse(mean_angle > pi, 
                       mean_angle - 2 * pi, 
                       mean_angle)
  predicted_angles <- c(predicted_angles, theta)
}

# Optionally convert to DOY
predicted_doy <- radians_to_days(predicted_angles)

frame<- data.frame(angles=predicted_angles, leafing=leafing_vals)

windows()
ggplot() +
  geom_point(data = df, aes(x = leafing, y = theta), color = "black") +
  geom_line(data = frame, aes(x = leafing, y = angles), color = "red") +
  theme_minimal()


B01 <- model$beta1[,"(Intercept)"]
B02 <- model$beta2[,"(Intercept)"]

a1 <- model$beta1[,"leafing"]
a2 <- model$beta2[,"leafing"]
  
aM <- atan2(B02+a2, B01+a1)
B<- atan2(B02,B01)
windows()
hist(aM)

windows()
hist(B)

new_data <- data.frame(leafing = seq(0, 100, length.out = 200))
predicted <- fitted(model, newdata = new_data, type = "link")

# Convert circular mean from radians to DOY
predicted$DOY <- (predicted$Circular.Mean * 365) / (2 * pi)



## model with bprms 

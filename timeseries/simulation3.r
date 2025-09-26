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
    prior(normal(25, 15), nlpar = "td"),
    prior(normal(50, 15), nlpar = "tf")
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
library(pacman)
p_load(dclone, MASS, ggplot2, snow, tidyverse)

n.years  <- 7
one.year <- seq(from=1,to=365,by=30)
samp.days <- rep(one.year,n.years)
n.inds <- 10
all.days <- rep(samp.days,n.inds)
n <- length(all.days)
year.id  <- rep(rep(1:n.years, each = length(one.year)), n.inds)
indv.id  <- rep(1:n.inds, each = length(samp.days))

logit.pf <- function(kd,Td,x){
  out <- kd*(x-Td)
  return(out)
}

sigsq <- 0.45  #noise levels 
kd <- 0.1
Td <- 100
mu.true <- logit.pf(kd=kd,Td=Td,x=all.days)

norm.samps <- rnorm(n=n, mean=mu.true, sd=sqrt(sigsq))
y.sims <- 1/(1+exp(norm.samps))

df<- data.frame(
  days=all.days,
  indv=indv.id,
  year=year.id,
  y=y.sims
)

windows()
ggplot(df, aes(x=days, y=y, color=as.factor(year))) +
  geom_point() +
  labs(title="Simulated phenology data",
       y="Simulated y",
       x="Day of year") +
  theme_minimal()


##JAGS model for intercept
leaves <- function(){
  lkd ~ dnorm(0,0.4)
  kd <- exp(lkd)
  ltd ~ dnorm(0,4)
  Td <- exp(ltd)
  ls ~ dnorm(0,1)
  sigsq <- pow(exp(ls),2)
  for(j in 1:n){
    muf[j] <-  kd*(days[j]-Td)
  }
  for(k in 1:K){
    for(i in 1:n){
      X[i,k] ~ dnorm(muf[i],1/sigsq)
    } 
  }
}



test.data <- log(1-y.sims) - log(y.sims)
data4dclone <- list(K=1, X=dcdim(data.matrix(test.data)), n=n, days=all.days)

cl.seq <- c(1,4,8,16);
n.iter<-10000;n.adapt<-5000;n.update<-100;thin<-10;n.chains<-3;

out.parms <- c("kd", "Td", "sigsq")
leaves.dclone <- dc.fit(data4dclone, params=out.parms, model=leaves, n.clones=cl.seq,
                        multiply="K",unchanged="n",
                        n.chains = n.chains, 
                        n.adapt=n.adapt, 
                        n.update=n.update,
                        n.iter = n.iter, 
                        thin=thin,
                        inits=list(lkd=log(0.2), ltd=log(40))
                        )

dcdiag(leaves.dclone)
summary(leaves.dclone)

dctable <- dctable(leaves.dclone)
windows()
plot(dctable)
windows()
plot(dctable, type="log.var")


#lets try real data with the same model


data<- read.csv("C:\\Users\\Vicente\\repo\\ForestLandscapes\\timeseries\\dataset_extracted\\cavallinesia.csv")
head(data)

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



trees<- unique(data$tree)
years<- unique(data$pheno_year)
all_before_threshold <- data.frame()
for (i in 1:length(trees)) {
  for (j in 1:length(years)) {
    subset_data <- data %>% filter(tree == trees[i], pheno_year == years[j])
    subset_data<- subset_data %>% arrange(day)
    windows()
    plot(subset_data$day, subset_data$y_norm, type='l', main=paste("Tree:", trees[i], "Year:", years[j]),
         xlab="Day", ylab="Normalized Leafing")
    #find values below threshold
    threshold <- 0.2
    below_threshold <- which(subset_data$y_norm < threshold)
    last_item<-max(below_threshold)
    #subset all values before last_item
    if (length(below_threshold) > 0 && last_item < nrow(subset_data)) {
      subset_before_threshold <- subset_data[1:last_item, ]
      points(subset_before_threshold$day, subset_before_threshold$y_norm, col='red', pch=19)
      all_before_threshold <- bind_rows(all_before_threshold, subset_before_threshold)
    }
      
  }}



windows()
ggplot(all_before_threshold, aes(x=day, y=y_norm, color=as.factor(tree_year))) +
  geom_point() +
  labs(title="Cavallinesia phenology data",
       y="Predicted leafing",
       x="Day of year") +
  theme_minimal()

#logit transform 
test.data <- log(1-all_before_threshold$y_norm) - log(all_before_threshold$y_norm)
data4dclone <- list(K=1, X=dcdim(data.matrix(test.data)), n=nrow(all_before_threshold), days=all_before_threshold$day)
cl.seq <- c(1,4,8,16);
n.iter<-10000;n.adapt<-5000;n.update<-100;thin<-10;n.chains<-3;
out.parms <- c("kd", "Td", "sigsq")
cava.intercept <- dc.fit(data4dclone, params=out.parms, model=leaves, n.clones=cl.seq,
                        multiply="K",unchanged="n",
                        n.chains = n.chains, 
                        n.adapt=n.adapt, 
                        n.update=n.update,
                        n.iter = n.iter, 
                        thin=thin,
                        inits=list(lkd=log(0.2), ltd=log(40))
)


summary(cava.intercept)

dcdiag(cava.intercept)
cavatable <- dctable(cava.intercept)
windows()
plot(cavatable)
windows()
plot(cavatable, type="log.var")
logit.pf(kd=coef(cava.intercept)["kd"],Td=coef(cava.intercept)["Td"],x=seq(1,365,1))->LC_values
LC_values<- 1/(1+exp(LC_values))

x <- seq(1, 365)

calendar_doy <- ((x + 243 - 1) %% 365) + 1

results_df <- data.frame(
  x = x,
  y = LC_values,
  doy = calendar_doy,
  date= as.Date(calendar_doy - 1, origin = "2018-01-01")
)
head(results_df)

#what date is 90%
results_df %>%
  filter(y <= 0.9) %>%
  slice(1)

results_df %>%
  filter(y <= 0.5) %>%
  slice(1)

results_df %>%
  filter(y <= 0.1) %>%
  slice(1)

day_date<- seq(from=as.Date("2018-09-01"), to=as.Date("2019-08-31"), by="month")
day_date_lookup <- data.frame(
  day = as.numeric(difftime(day_date, as.Date("2018-09-01"), units = "days")),
  date = day_date
)


windows()
ggplot(all_before_threshold,
       aes(x = day, y = y_norm, color = as.factor(tree_year))) +
  geom_point() +
  geom_line(data = results_df, aes(x = x, y = y),
            color = "black", linewidth = 1) +
  scale_x_continuous(
    breaks = day_date_lookup$day,
    labels = format(day_date_lookup$date, "%b %d")
  ) +
  labs(
    title = "Cavallinesia phenology data",
    y = "Predicted leafing",
    x = "Date"
  ) +theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  geom_vline(xintercept = c(87,126, 165), linetype = "dashed", color = "black")
  

#lets show it circularly
data<- data %>%
  mutate(
    is_leap = if_else(year %% 4 == 0 & (year %% 100 != 0 | year %% 400 == 0), TRUE, FALSE),
    rad = if_else(is_leap, 2 * pi * DOY / 366, 2 * pi * DOY / 365)
  )

data$month <- factor(data$month, levels = 1:12, labels = month.abb)

windows()
ggplot(data, aes(x = month, y = 1-y_norm)) +
  geom_boxplot()+
  coord_polar() 

# Simulate data
one.year <- seq(from = 1, to = 365, by = 30)
n.years  <- 7
samp.days <- rep(one.year, n.years)
n.inds   <- 20
all.days <- rep(samp.days, n.inds)
n        <- length(all.days)
year.id  <- rep(rep(1:n.years, each = length(one.year)), n.inds)
indv.id  <- rep(1:n.inds, each = length(samp.days))

pf_fun <- function(kd, Td, x) {
  1 / (1 + exp(kd * (x - Td)))
}

b  <- 20
kd <- 0.1
Td <- 150

# Generate year effects on Td
uY_true    <- rep(0, n.years)
uI_true    <- rep(0, n.inds)
Td_year_indv<- Td + uY_true[year.id] + uI_true[indv.id]       


# simulate samples
pf_true  <- pf_fun(kd = kd, Td = Td_year_indv, x = all.days)
a_true   <- (pf_true * b) / (1 - pf_true)
beta.samps <- rbeta(n = n, shape1 = a_true, shape2 = b)

# overlay both series in one plot
df<- data.frame(
  days = all.days,
  year = as.factor(year.id),
  indv = as.factor(indv.id),
  beta = beta.samps,
  pf   = pf_true
)
windows()
ggplot(df[df$indv %in% c(1,2,3), ], aes(x = days, y = beta, color = indv)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = pf), size = 1)+
  labs(title = "Simulated beta samples and true phenology curves",
       y = "Beta samples and true pf",
       x = "Day of year") +
  theme_minimal()

     
leaves <- function() {
  lkd ~ dnorm(pr[1,1], 1 / pow(pr[1,2],2))
  kd <- exp(lkd)
  ltd ~ dnorm(pr[2,1], 1 / pow(pr[2,2],2))
  Td <- exp(ltd)
  lb ~ dnorm(pr[3,1], 1 / pow(pr[3,2],2))
  b <- exp(lb)

  for (j in 1:n) {
    pf[j] <- 1 / (1 + exp(kd * (days[j] - Td)))
    a[j]  <- (pf[j] * b) / (1 - pf[j])
  }
  for (k in 1:K) {
    for (i in 1:n) {
      Y[i,k] ~ dbeta(a[i], b)
    }
  }
}


upfun_leaves <- function(x) {
  if (missing(x)) {
    means <- c(log(0.1), log(150), log(20))
    sds   <- c(2, 2, 2)
    return(cbind(means, sds))
  }

  ncl <- nclones(x) # When the function kicks in with another k cloning step greater than 1. So the function says how many clones are there in this iteration
  if (is.null(ncl)) ncl <- 1   # of course, if no cloning yet, ncl=1

  par <- coef(x)  #we return the posterior means of the parameters GIVEN BY OUT.PARAMS
  se  <- dcsd(x)  #we return the posterior SDs of the parameters GIVEN BY OUT.PARAMS

  needed <- c("lkd", "ltd", "lb")  # needed parameters, only priors
  missing_pars <- setdiff(needed, names(par))
  if (length(missing_pars) > 0) {
    stop("upfun_leaves requires monitored parameters: ",
         paste(missing_pars, collapse = ", "))
  }
  means <- par[needed]   
  sds   <- se[needed] * sqrt(ncl)  # multiply by sqrt(ncl) to get un-cloned SDs
  return(cbind(means, sds))
}


dat<- list(K=1,                               #K is a loop index for data cloning
           Y=dcdim(data.matrix(beta.samps)),  
           n=n,
           days=all.days,
           pr = upfun_leaves())
cl.seq <- c(1,5,10,20); # flone sequence
n.iter<-1000;n.adapt<-500;n.update<-10;thin<-1;n.chains<-3;
out.parms <- c("lkd", "ltd", "lb", "kd", "Td", "b")
leaves.dclone <- dc.fit(dat, params=out.parms, model=leaves, n.clones=cl.seq,
                        multiply="K",unchanged=c("n"),
                        n.chains = n.chains, 
                        n.adapt=n.adapt, 
                        n.update=n.update,
                        n.iter = n.iter,
                        update= "pr",
                        updatefun=upfun_leaves,
                        thin=thin)

dcdiag(leaves.dclone)
summary(leaves.dclone)

dcTable <- dctable(leaves.dclone)
windows()
plot(dcTable)
plot(dcTable, type="log.var")
windows()
plot(dcTable)

mcmc_list <- as.mcmc(leaves.dclone) 
mcmcapply(leaves.dclone, sd) * sqrt(nclones(leaves.dclone))

coef(leaves.dclone)



leaves2 <- function() {
  lkd ~ dnorm(pr[1,1], 1 / pow(pr[1,2], 2))
  kd <- exp(lkd)
  ltd ~ dnorm(pr[2,1], 1 / pow(pr[2,2], 2))
  Td <- exp(ltd)
  lb ~ dnorm(pr[3,1], 1 / pow(pr[3,2], 2))
  b  <- exp(lb)

  log.sigmaY ~ dnorm(pr[4,1], 1 / pow(pr[4,2], 2))
  sigmaY <- exp(log.sigmaY)
  tauY   <- pow(sigmaY, -2)

  for (y in 1:nyear) {
  zY[y] ~ dnorm(0,1)
  uY[y] <- sigmaY * zY[y]
  }

  for (i in 1:n) {
    ltd_y[i] <- ltd + uY[year[i]]   # log-scale Td for obs i
    Td_y[i]  <- exp(ltd_y[i])
    pf[i]    <- 1 / (1 + exp(kd * (days[i] - Td_y[i])))
    a[i]     <- (pf[i] * b) / (1 - pf[i])
  }

  for (k in 1:K) {
    for (i in 1:n) {
      Y[i,k] ~ dbeta(a[i], b)
    }
  }
}

upfun_leaves2 <- function(x) {
  if (missing(x)) {
    init_means <- c(
      lkd = log(0.1),
      ltd = log(150),
      lb  = log(20),
      log.sigmaY = log(1)
    )

    init_sds <- c(
      lkd = 1.5,
      ltd = 2,
      lb  = 2,
      log.sigmaY = 1.5
    )
    return(cbind(init_means, init_sds))
  }

  ncl <- nclones(x)
  if (is.null(ncl)) ncl <- 1

  par <- coef(x)
  se  <- dcsd(x)

  needed <- c("lkd", "ltd", "lb", "log.sigmaY")
  missing_pars <- setdiff(needed, names(par))
  if (length(missing_pars) > 0) {
    stop("upfun_leaves2 requires monitored parameters: ",
         paste(missing_pars, collapse = ", "))
  }

  means <- par[needed]
  sds <- se[needed] * sqrt(ncl)

  return(cbind(means, sds))
}


Ymat <- dcdim(data.matrix(beta.samps))  # yields n x 1 initially; dc.fit will properly expand by multiply

data4dclone <- list(
  K     = 1,            # number of clones along Y dimension
  Y     = Ymat,         # n x K (dcdim object works with multiply="K")
  n     = n,            # number of obs (e.g. 1750)
  days  = all.days,     # length n
  nyear = n.years,      # number of years (e.g. 7)
  year  = year.id,      # vector of length n with integers 1:nyear
  pr    = upfun_leaves2()  # 4 x 2 matrix of (means, SDs)
)

# Parameters to monitor (with priors)
out.parms <- c("lkd", "ltd", "lb", "log.sigmaY", "Td", "kd", "b", "uY")

cl <- makePSOCKcluster(3)
# Run data cloning: clone along n (observation index)
leaves2.dclone <- dc.parfit(
  cl         = cl,
  data       = data4dclone,
  params     = out.parms,
  model      = leaves2,
  n.clones   = c(1, 5, 10, 20),
  multiply   = "K",      # clone acrsoss the K dimension (columns)
  unchanged  = c("n","K","nyear","pr"),      # n (days, year) don't change
  n.chains   = 3,
  n.adapt    = 500,
  n.update   = 100,
  n.iter     = 10000,
  thin       = 10,
  partype    = "parchains",   
  updatefun  = upfun_leaves2,
  update     = "pr"
)



summary(leaves2.dclone)
coef(leaves2.dclone)

dcdiag(leaves2.dclone)

coef(leaves2.dclone)["ltd"] + coef(leaves2.dclone)["uY[1]"]
exp(coef(leaves2.dclone)["ltd"] + coef(leaves2.dclone)["uY[1]"])
exp(coef(leaves2.dclone)["ltd"] + coef(leaves2.dclone)["uY[2]"]) 

dctable<- dctable(leaves2.dclone)

windows()
plot(dctable, which= 1:6, type="log.var")

length(dctable)
mcmc_list2


## trying to add more random effects: individual-level effects
leaves3 <- function() {
  lkd ~ dnorm(pr[1,1], 1 / pow(pr[1,2], 2))
  kd  <- exp(lkd)
  ltd ~ dnorm(pr[2,1], 1 / pow(pr[2,2], 2))
  lb ~ dnorm(pr[3,1], 1 / pow(pr[3,2], 2))
  b  <- exp(lb)

  log.sigmaY ~ dnorm(pr[4,1], 1 / pow(pr[4,2], 2))
  sigmaY <- exp(log.sigmaY)
  tauY   <- pow(sigmaY, -2)

  log.sigmaI ~ dnorm(pr[5,1], 1 / pow(pr[5,2], 2))
  sigmaI <- exp(log.sigmaI)
  tauI   <- pow(sigmaI, -2)

  for (y in 1:nyear) {
    uY[y] ~ dnorm(0, tauY)
  }

  for (j in 1:nind) {
    uI[j] ~ dnorm(0, tauI)
  }

  for (i in 1:n) {
    ltd_y[i] <- ltd + uY[year[i]] + uI[indiv[i]]   # still log-scale
    Td_y[i]  <- exp(ltd_y[i])
    pf[i]    <- 1 / (1 + exp(kd * (days[i] - Td_y[i])))
    a[i]     <- (pf[i] * b) / (1 - pf[i])
  }

  for (k in 1:K) {
    for (i in 1:n) {
      Y[i,k] ~ dbeta(a[i], b)
    }
  }
}

upfun_leaves3 <- function(x) {
  if (missing(x)) {
    init_means <- c(
      lkd = log(0.1),
      ltd = log(150),
      lb  = log(20),
      log.sigmaY = log(1),
      log.sigmaI = log(1)    # initial guess for individual SD
    )

    init_sds <- c(
      lkd = 1.5,
      ltd = 2,
      lb  = 2,
      log.sigmaY = 1.5,
      log.sigmaI = 1.5
    )
    return(cbind(init_means, init_sds))
  }

  ncl <- nclones(x)
  if (is.null(ncl)) ncl <- 1

  par <- coef(x)
  se  <- dcsd(x)

  needed <- c("lkd", "ltd", "lb", "log.sigmaY", "log.sigmaI")
  missing_pars <- setdiff(needed, names(par))
  if (length(missing_pars) > 0) {
    stop("upfun_leaves3 requires monitored parameters: ",
         paste(missing_pars, collapse = ", "))
  }

  means <- par[needed]
  sds <- se[needed] * sqrt(ncl)

  return(cbind(means, sds))
}

Ymat <- dcdim(data.matrix(beta.samps))  # yields n x 1 initially; dc.fit will properly expand by multiply

data4dclone3 <- list(
  K     = 1,            # number of clones along Y dimension
  Y     = Ymat,         # n x K (dcdim object works with multiply="K")
  n     = n,            # number of obs (e.g. 1750)
  nind  = n.inds,      # number of individuals
  days  = all.days,     # length n
  nyear = n.years,      # number of years (e.g. 7)
  year  = year.id,      # vector of length n with integers 1:nyear
  indiv = indv.id,      # vector of length n with integers 1:nind
  pr    = upfun_leaves3()  # 5 x 2 matrix of (means, SDs)
)

# Parameters to monitor (with priors)
out.parms <- c("lkd", "ltd", "lb", "log.sigmaY", "Td", "kd", "b", "uY")

cl <- makePSOCKcluster(3)
# Run data cloning: clone along n (observation index)
leaves3.dclone <- dc.parfit(
  cl         = cl,
  data       = data4dclone,
  params     = out.parms,
  model      = leaves3,
  n.clones   = c(1, 5, 10),
  multiply   = "K",      # clone acrsoss the K dimension (columns)
  unchanged  = c("n","K","nyear","pr","nind"),      # n (days, year) don't change
  n.chains   = 3,
  n.adapt    = 50,
  n.update   = 10,
  n.iter     = 1000,
  thin       = 1,
  partype    = "parchains",   
  updatefun  = upfun_leaves3,
  update     = "pr"
)

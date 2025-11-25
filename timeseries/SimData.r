library(pacman)
p_load(dclone, MASS, ggplot2)

# Simulate data
one.year <- seq(from = 1, to = 365, by = 30)
n.years  <- 7
samp.days <- rep(one.year, n.years)
n.inds   <- 10
all.days <- rep(samp.days, n.inds)
n        <- length(all.days)
year.id  <- rep(rep(1:n.years, each = length(one.year)), n.inds)

pf_fun <- function(kd, Td, x) {
  1 / (1 + exp(kd * (x - Td)))
}

b  <- 20
kd <- 0.1
Td <- 150

# Generate year effects on Td
uY_true    <- rep(0, n.years)
uY_true[1] <- 30        
Td_year <- Td + uY_true[year.id]

# simulate samples
pf_true  <- pf_fun(kd = kd, Td = Td_year, x = all.days)
a_true   <- (pf_true * b) / (1 - pf_true)
beta.samps <- rbeta(n = n, shape1 = a_true, shape2 = b)

# overlay both series in one plot
cols <- rainbow(n.years)
windows()
plot(all.days, beta.samps,
  col = cols[year.id],
  pch = 16,
  main = "Simulated beta data with year-1 effect",
  xlab = "Day of year", ylab = "Beta response",
  ylim = c(0, 1))

# Plot the true pf curve as lines (one line per year, matching the point colors)
for (j in 1:n.years) {
  days_j <- one.year
  Td_j <- Td + uY_true[j]
  pf_j <- pf_fun(kd = kd, Td = Td_j, x = days_j)
  lines(days_j, pf_j, col = cols[j], lwd = 2)
}
legend("topright", legend = paste("Year", 1:n.years), col = cols, pch = 16, bty = "n")

     
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
    uY[y] ~ dnorm(0, tauY)
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

# Run data cloning: clone along n (observation index)
leaves2.dclone <- dc.fit(
  data       = data4dclone,
  params     = out.parms,
  model      = leaves2,
  n.clones   = c(1, 5),
  multiply   = "K",      # clone acrsoss the K dimension (columns)
  unchanged  = c("n","K","nyear","pr"),      # n (days, year) don't change
  n.chains   = 3,
  n.adapt    = 500,
  n.update   = 10,
  n.iter     = 1000,
  thin       = 1,
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
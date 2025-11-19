library(pacman)
library(MASS)
p_load(MASS,dclone, mcmcplots, ggplot2)
# R functions for Data Cloning (maximum likelihood estimation using Bayesian MCMC)
library(dclone); 
# Create plots for MCMC output
library(mcmcplots)
library(ggplot2)


# Simulate data
one.year <- seq(from=1,to=365,by=30)
n.years <- 7
samp.days <- rep(one.year,n.years)
n.inds <- 25
all.days <- rep(samp.days,n.inds)
n <- length(all.days)

pf <- function(kd,Td,x){
  out <- 1/(1+exp(kd*(x-Td)))
  return(out)
}

#####visualize shpe of the generation function###
tmp<-pf(kd=0.9, Td=100, x=all.days)
windows()
ggplot(data=data.frame(x=all.days,y=pf.true), aes(x=x,y=y)) + geom_point()
#########################################

b <- 20
kd <- 0.1
Td <- 50
pf.true <- pf(kd=kd,Td=Td,x=all.days)

#plot(all.days, pf.true, pch=16)
as <- (pf.true*b)/(1-pf.true)

beta.samps <- rbeta(n=n, shape1=as, shape2=b)

windows()
plot(all.days,beta.samps)


leaves <- function() {

  lkd ~ dnorm(0, 0.4)    # prior for lkd
  kd <- exp(lkd)

  ltd ~ dnorm(0, 4)      # prior for ltd
  Td <- exp(ltd)

  lb ~ dnorm(0, 1)       # prior for lb
  b <- exp(lb)

  # main loop: add the year effect on Td (your formulation)
  for (j in 1:n) {
    pf[j] <- 1/(1 + exp(kd * (days[j] - Td)))
    a[j]  <- (pf[j] * b) / (1 - pf[j])
  }

  # data cloning loop
  for (k in 1:K) {
    for (i in 1:n) {
      Y[i,k] ~ dbeta(a[i], b)
    }
  }
}

leaves2 <- function() {

  lkd ~ dnorm(0, 0.4)    # prior for lkd
  kd <- exp(lkd)

  ltd ~ dnorm(0, 4)      # prior for ltd
  Td <- exp(ltd)

  lb ~ dnorm(0, 1)       # prior for lb
  b <- exp(lb)

  # year random effects
  for (y in 1:years) {
    uY[y] ~ dnorm(0, tauY)
  }
  sigmaY ~ dunif(0, 10)
  tauY <- pow(sigmaY, -2)

  # main loop: add the year effect on Td (your formulation)
  for (j in 1:n) {
    pf[j] <- 1/(1 + exp(kd * (days[j] - (Td + uY[year[j]]))))
    a[j]  <- (pf[j] * b) / (1 - pf[j])
  }

  # data cloning loop
  for (k in 1:K) {
    for (i in 1:n) {
      Y[i,k] ~ dbeta(a[i], b)
    }
  }
}

library(dclone)
data4dclone <- list(K=1, Y=dcdim(data.matrix(beta.samps)), n=n, days=all.days) # just creates a list

cl.seq <- c(1,5,10);   # is this the clones sequence?
n.iter<-1000;n.adapt<-500;n.update<-10;thin<-1;n.chains<-3;

out.parms <- c("kd", "Td", "b", "pf")
leaves.dclone <- dc.fit(data4dclone, params=out.parms, model=leaves, n.clones=cl.seq,
                        multiply="K",unchanged="n",
                        n.chains = n.chains, 
                        n.adapt=n.adapt, 
                        n.update=n.update,
                        n.iter = n.iter, 
                        thin=thin)


dcdiag(leaves.dclone)

# Now create "leaves2.0" where you add a random effect on Td

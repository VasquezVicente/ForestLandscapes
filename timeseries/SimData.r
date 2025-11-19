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

v.years <- rep(1:n.years, each=length(one.year)*n.inds)
data4dclone <- list(K=1, Y=dcdim(data.matrix(beta.samps)), n=n, days=all.days, year= v.years) # just creates a list

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


##understanding the dataclonig
set.seed(1234)
n<-50
beta<-c (1.8,-0.9)
sigma<- 0.2
x<- runif(n,0,1)
X<- model.matrix(~x)
alpha<- rnorm(n, mean=0, sd=sigma)
lambda<- exp(alpha+drop(X%*%beta))
Y<- rpois(n, lambda)

dat<- list(Y=Y, X=X, n=n, np=ncol(X))

glmm.model.up<- function(){
  for (i in 1:n){
    Y[i] ~ dpois(lambda[i])
    lambda[i] <- exp(alpha[i]+ inprod(X[i,], beta[1,]))
    alpha[i] ~ dnorm(0, 1/sigma^2)
  }

  for (j in 1:np){
    beta[1,j] ~ dnorm(pr[j,1], pr[j,2])
  }
  log.sigma ~ dnorm(pr[(np+1),1], pr[(np+1),2])
  sigma <- exp(log.sigma)
  tau <- 1/ pow(sigma, 2)
}

upfun <- function(x) {
    if (missing(x)) {
       np <- ncol(X)
       return(cbind(rep(0, np+1),
           rep(0.001, np+1)))
    } else {
       ncl <- nclones(x)
       if (is.null(ncl))
          ncl <- 1
       par <- coef(x)
       se <- dcsd(x)
       log.sigma <- mcmcapply(x[,"sigma"], log)
       par[length(par)] <- mean(log.sigma)
       se[length(se)] <- sd(log.sigma) * sqrt(ncl)
       return(cbind(par, se))
    }
 }

mod<- jags.fit(dat, c("beta","sigma"), glmm.model, n.iter=1000)

summary(mod)
windows()
plot(mod)

#the cloning functions

dclone(1:5,1) #clone it once
dclone(1:5,2) # returns a list wit 1,2,3,4,5,1,2,3,4,5
dclone(matrix(1:4,2,2),2) # clone a matrix one in top of the other
dclone(data.frame(a=1:2,b=3:4),2) # clone a data frame one on top of the other

dat2<- dclone(dat, n.clones=2, multiply="n", unchanged="np") #atrribute np is number of columns of X which is not cloned

mod2<- jags.fit(dat2, c("beta","sigma"), glmm.model, n.iter=1000)
summary(mod2)
summary(mod)


# extra dimensions for timeseries and autoregressive models

obj<- dclone(dcdim(data.matrix(1:5)),2)


beverton.holt<- function(){
  for (j in 1:k) {
    for (i in 2:(n+1)) {
      Y[(i-1),j] ~ dpois(exp(log.N[i,j]))
      log.N[i,j] ~ dnorm(mu[i,j], 1 / sigma^2)
      mu[i,j]<- log(lambda) + log.N[(i-1),j]
        -log(1+beta*exp(log.N[(i-1),j]))
    }
    log.N[1,j] ~ dnorm(mu0, 1 / sigma^2)
    }
  beta ~ dlnorm(-1,1)
  sigma ~ dlnorm(0,1)
  tmp ~ dlnorm(0,1)
  lambda<- tmp + 1
  mu0<- log(lambda) + log(2) - log(1+ beta*2)
}

paurelia <- c(17, 29, 39, 63, 185, 258, 267, 392, 510, 570, 650, 560, 575, 650, 550, 480, 520, 500)
bhdat <- list(Y=dcdim(data.matrix(paurelia)), n=length(paurelia), k=1)
dcbhdat <- dclone(bhdat, n.clones = 5,multiply = "k", unchanged = "n")

bhmod <- jags.fit(dcbhdat, 
    c("lambda","beta","sigma"), beverton.holt, 
    n.iter=1000)

coef(bhmod)
summary(bhmod)


updat<- list(Y=Y, X=X, n=n, np=ncol(X), pr= upfun())
k<- c(1,5,10,20)
dcmod<- dc.fit(updat, params=c("beta","sigma"), model=glmm.model.up,
               n.clones=k, multiply="n", unchanged="np",
               n.iter=1000,upfun=upfun)

summary(dcmod)
dct<- dctable(dcmod)
plot(dct)

plot(dct, type="log.var")

dcdiag(dcmod)

coef(dcmod)
dcsd(dcmod)
mcmcapply(dcmod,sd) * sqrt(nclones(dcmod))

confint(dcmod)


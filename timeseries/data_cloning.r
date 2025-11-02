#data cloning
#single leaf drop estimation
library(pacman)
p_load(dclone,ggplot2)

#example with posisant
sigma<- 0.2
beta<- c(1.8, -0.9)
n<-50
x<-runif(n,0,1)
X<-model.matrix(~x)
alpha<- rnorm(n, mean=0, sd=sigma)
lambda<- exp(alpha + drop(X %*% beta))

Y<- rpois(n, lambda)

df<- data.frame(x=X[,2], Y=Y)

windows()
ggplot(df, aes(x=x, y=Y)) + geom_point() +
  stat_smooth(method="glm", method.args=list(family="poisson"), se=FALSE, color="red")


glmm.model <- function() {
   for (i in 1:n) {   # so for i 1 through n 
       Y[i] ~ dpois(lambda[i])   # simulate Y as a Poisson with mean lambda but we are misisng how many ns?
       lambda[i] <- exp(alpha[i] +  # here come the lambda with exp to get the log, plus the Beta time the observations 
          inprod(X[i,], beta[1,]))  #what the hell is inprod?
       alpha[i] ~ dnorm(0, tau)   # alpha is distributed normally with mean 0 and precision tau, hell how do i know is 0 and normally distributed?
    }
    for (j in 1:np) {
       beta[1,j] ~ dnorm(0, 0.001)  # now beta is what we are estimating? and we also magically know is at 0, well thats just a prior 
    }
    log.sigma ~ dnorm(0, 0.001)  # the log of sigma  thats a prior i think 
    sigma <- exp(log.sigma)      # then we exp because thats the prior of sigma
    tau <- 1 / pow(sigma, 2)    # tau is 1 over sigma squared, where the hell did tau came from 
}

dat <- list(Y = Y, X = X, n = n, np = ncol(X))
dat2<- dclone(dat, n.clones=2, multiply="n", unchanged= "np")

mod2 <- jags.fit(dat2,
    c("beta", "sigma"), glmm.model, n.iter = 1000)


summary(mod2)



## gaussian processes from scratch
## 
## combining some code from
## https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/master/gaussian_processes.ipynb
## and equations from Rasmussen & Williams as reference (see www.GaussianProcess.org/gpml)
## see https://github.com/krasserm/bayesian-machine-learning for repo

rm(list = ls())
library(data.table)
library(magrittr)
library(ggplot2)
theme_set(theme_bw())

###### GP ML -------------------------------------------------------------------
## using Rasmussen & Williams as reference (see www.GaussianProcess.org/gpml)
## also some code from https://gist.github.com/jkeirstead/2312411
## covariance function --> squared exponential (eq. 2.16)
## but here we also add the two hyperparameters, l and sigma_f (length and height)
#' @param X1 matrix (m,d) where d is the number of covariates
#' @param X2 matrix (n,d) 
#' @param l  characteristic length. default = 1
#' @param sigma_f noise var default = 1
#' @description squared exp kernel given by k(x_p, x_q) = sigma_f^2*exp(-1/(2l^2))
calcSigmaMat <- function(X1, X2, l=1, sigma_f = 1){
  Sigma <- matrix(0, nrow = nrow(X1), ncol = nrow(X2))
  for (iR in 1:nrow(X1)){
    for (iC in 1:nrow(X2)){
      sqDist <- X1[iR,] - X2[iC, ]
      sqDist1 <- t(sqDist) %*% sqDist
      Sigma[iR, iC] <- sigma_f^2*exp(-sqDist1/(2*l^2))
    }
  }
  return(Sigma)
}


## generate multinorm samples
sample_mvnorm <- function(n_samples, mu, sigma, x){
  library(data.table)
  library(magrittr)
  
  samples <- MASS::mvrnorm(n_samples, mu, sigma)
  
  ## return as data table
  output <- as.data.table(t(samples)) %>% 
    .[, x := x] %>% 
    .[, mu := mu] %>% 
    melt.data.table(., id.vars = c("x", "mu"))
  return(output)
}


## predictive posterior 
posterior_pred <- function(x_new, x_train, y_train, l=1, sigma_f=1, sigma_n=1e-8){
  
  ## calculate the covariance matrix
  # k <- calcSigmaMat(x_train, x_train, l, sigma_f)
  
  ## now we calculate the various k matrices
  k_xx   <- calcSigmaMat(x_train, x_train, l, sigma_f)
  k_xxs  <- calcSigmaMat(x_train, x_new,   l, sigma_f)
  k_xsx  <- calcSigmaMat(x_new,   x_train, l, sigma_f)
  k_xsxs <- calcSigmaMat(x_new,   x_new,   l, sigma_f)
  
  ## and the matrix inverses
  kInv <- solve(k_xx + sigma_n*diag(nrow(k_xx)))
  predMean <- k_xsx %*% kInv %*% y_train
  predCov <- k_xsxs - k_xsx %*% kInv %*% k_xxs
  
  ## the marginal likelihood
  marginal_llk <- -0.5*t(y_train) %*% kInv %*% y_train - 
    0.5*log(det(k_xx + sigma_n*diag(nrow(k_xx)))) - 
    nrow(x_train)/2*log(2*pi)
  
  output <- list(predMean = predMean, 
                 predCov = predCov,
                 marginal_llk = marginal_llk)
  return(output)
}

## Simple examples -------------------------------------------------------------
## replicate https://gist.github.com/jkeirstead/2312411 but we use the 
## function above to handle multivariate case
nSamples <- 25
sigma_f <- 1
seedVal <- 202003

## generate some initial input data
x.star <- as.matrix(seq(-5, 5, len=nSamples))
covMat <- calcSigmaMat(x.star, x.star)

# Generate a number of functions from the process
set.seed(seedVal)
nFunctionSamples <- 3
priorSamples <- sample_mvnorm(nFunctionSamples, 
                              rep(0, length(x.star)), 
                              covMat, 
                              x.star)

## plot the samples
ggplot(priorSamples, aes(x=x, y=value)) +
  geom_line(aes(group=variable), lty = 2) +
  theme_bw() +
  xlab("input, x")


# 2. Now let's assume that we have some known data points;
# this is the case of Figure 2.2(b). In the book, the notation 'f'
# is used for f$y below.  I've done this to make the ggplot code
# easier later on.
f <- data.table(x=c(-4,-3,-1,0,2),
                y=c(-2,0,1,2,-1))

# Calculate the covariance matrices
# using the same x.star values as above
x <- as.matrix(f$x)
k.xx <- calcSigmaMat(x,x)
k.xxs <- calcSigmaMat(x,x.star)
k.xsx <- calcSigmaMat(x.star,x)
k.xsxs <- calcSigmaMat(x.star,x.star)


## eq 2.22 to 2.24
## the conditional distribution of the new data 
## the noiseless case is a special case where sigma_n = 0
sigma_n <- 0 # the noiseless case
kInv <- solve(k.xx + sigma_n*diag(nrow(k.xx)))
predMean <- k.xsx %*% kInv %*% as.matrix(f$y)
predCov <- k.xsxs - k.xsx %*% kInv %*% k.xxs

## generate samples of the conditional distribution
predSamples <- sample_mvnorm(50, 
                             predMean, 
                             predCov, 
                             x.star)

ggplot() + 
  geom_line(data=predSamples, aes(x=x, y=value, group=variable), 
            colour = "grey50", alpha = 0.5) + 
  geom_point(data=f, aes(x=x, y=y)) +
  geom_line(data=NULL, aes(x=x.star, y=predMean), colour = "red") + 
  theme_bw()


## with noise
sigma_n <- 0.5
kInv <- solve(k.xx + sigma_n*diag(nrow(k.xx)))
predMeanNoise <- k.xsx %*% kInv %*% as.matrix(f$y)
predCovNoise <- k.xsxs - k.xsx %*% kInv %*% k.xxs

## generate samples of the conditional distribution
predSamplesNoise <- sample_mvnorm(100, 
                                  predMeanNoise, 
                                  predCovNoise, 
                                  x.star)

ggplot() + 
  geom_line(data=predSamplesNoise, aes(x=x, y=value, group=variable), 
            colour = "grey50", alpha = 0.2) + 
  geom_point(data=f, aes(x=x, y=y)) +
  geom_line(data=NULL, aes(x=x.star, y=predMeanNoise), colour = "red") + 
  theme_bw()


## Varying Hyperparameters -----------------------------------------------------
seedVal <- 202003
x_train <- matrix(c(-4,-3,-1,0,2), ncol = 1)
y_train <- matrix(c(-2,0,1,2,-1), ncol = 1)
set.seed(seedVal)
x_new <- matrix(runif(10, min = -15, max = 15), ncol = 1)
l <- 1
sigma_f <- 1
sigma_n <- 0

pred_post <- posterior_pred(x_new, x_train, y_train, l, sigma_f, sigma_n)

predSamplesNoise <- sample_mvnorm(100, 
                                  pred_post$predMean, 
                                  pred_post$predCov, 
                                  x_new)

ggplot() + 
  geom_line(data=predSamplesNoise, aes(x=x, y=value, group=variable), 
            colour = "grey50", alpha = 0.2) + 
  geom_line(data=NULL, aes(x=x_new, y=pred_post$predMean), colour = "red") + 
  geom_point(data=NULL, aes(x=x_train, y=y_train), colour = "blue") +
  theme_bw()


## vary sigma_f, l and sigma_n
vec_sigma_f <- c(1)
vec_l <- c(1, 3)
vec_sigma_n <- c(1e-8, 1)

all_output <- list()
all_samples <- list()
i <- 1
for (iSf in vec_sigma_f){
  for (iL in vec_l){
    for (iSn in vec_sigma_n){
      pred_post <- posterior_pred(x_new, x_train, y_train, iL, iSf, iSn)
      pred_samples <- sample_mvnorm(100, 
                                    pred_post$predMean, 
                                    pred_post$predCov, 
                                    x_new) %>% 
        .[, l := iL] %>% 
        .[, sigma_f := iSf] %>% 
        .[, sigma_n := iSn]
      all_output[[i]] <- pred_post
      all_samples[[i]] <- pred_samples
      i <- i + 1
    }
  }
}

allSamplesDf <- rbindlist(all_samples)

ggplot() + 
  geom_line(data=allSamplesDf, 
            aes(x=x, y=value, group=variable, colour = as.factor(sigma_n)),
            alpha = 0.2) +
  geom_point(data=allSamplesDf, aes(x=x, y=mu, colour = as.factor(sigma_n))) +
  geom_line(data=allSamplesDf, aes(x=x, y=mu, colour = as.factor(sigma_n))) + 
  facet_grid(l~sigma_n) + 
  theme(legend.position = "bottom") + 
  geom_point(data=data.frame(x=x_train, y=y_train), aes(x=x, y=y), colour = "blue") 


## add another data point 
seedVal <- 202003
x_train <- matrix(c(-8, -4,-3,-1,0,2,10), ncol = 1)
y_train <- matrix(c(2, -2,0,1,2,-1,3), ncol = 1)
set.seed(seedVal)
x_new <- matrix(runif(10, min = -15, max = 15), ncol = 1)

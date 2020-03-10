## GP with co2 concentration data 
## example in Rassmussen and code derived from 
## https://docs.pymc.io/notebooks/GP-MaunaLoa.html
## 
## using rstan instead of pymc3
rm(list=ls())
library(data.table)
library(magrittr)
library(lubridate)
library(stringr)
library(ggplot2)
theme_set(theme_bw())

## to download data
# co2Fid <- "https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv"
# 
# ## skip the first 58 lines as these are all fluff
# rawDat <- fread(co2Fid, skip = 58) %>% 
#   setnames(., c("yr", "mth", "date_excel", "date_num",
#                       "co2", "seasonally_adjusted", "fit", 
#                       "seasonally_adjusted_fit",
#                       "co2_filled", "seasonally_adjusted_filled")) 
# fwrite(rawDat, "./data/monthly_in_situ_co2_mlo.csv")

## read data from previously downloaded file
co2Fid <- "./data/monthly_in_situ_co2_mlo.csv"
rawDat <- fread(co2Fid)

## clean up 
cleanDat <- rawDat %>% 
  copy() %>% 
  ## drop the weird date cols
  .[, c("date_excel", "date_num") := NULL] %>% 
  ## replace -99.99 with NA
  .[, lapply(.SD, function(x) ifelse(x == -99.99, NA, x))] %>% 
  ## drop the NA rows
  na.omit(.) %>% 
  ## add date, make each month the 15th
  .[, dt := ymd(paste0(yr, "/", str_pad(mth, width = 2, "left", "0"), "/15"))] %>% 
  ## index -- divide by 365 days which is not exactly what you get in the 
  ## https://docs.pymc.io/notebooks/GP-MaunaLoa.html but close enough
  .[, dt_idx := as.numeric((dt - ymd('1958-03-15'))/365)] %>% 
  ## normalise co2
  .[, co2_norm := (co2 - co2[1])/sd(co2)] %>% 
  ## lastly add the test/train index --> use data up to end 2003 for training
  .[, isTrain := dt < ymd("2004-01-01")]

## plot
ggplot(cleanDat, aes(x = dt, y = co2)) + 
  geom_line()

ggplot(cleanDat, aes(x = dt_idx, y = co2_norm)) + 
  geom_line()

trainDat <- cleanDat[isTrain==TRUE]
testDat <- cleanDat[isTrain==FALSE]


## GP using R functions --------------------------------------------------------
## using Rasmussen & Williams as reference (see www.GaussianProcess.org/gpml)
## also some code from https://gist.github.com/jkeirstead/2312411
## covariance function --> squared exponential (eq. 2.16)
## but here we also add the two hyperparameters, l and sigma_f (length and height)
#' @param X1 matrix (m,d) where d is the number of covariates
#' @param X2 matrix (n,d) 
#' @param l  characteristic length. default = 1
#' @param sigma_f noise var default = 1
#' @description squared exp kernel given by k(x_p, x_q) = sigma_f^2*exp(-1/(2l^2))
calcK1Cov <- function(X1, X2, t1=1, t2=1){
  Sigma <- matrix(0, nrow = nrow(X1), ncol = nrow(X2))
  for (iR in 1:nrow(X1)){
    for (iC in 1:nrow(X2)){
      sqDist <- X1[iR,] - X2[iC, ]
      sqDist1 <- t(sqDist) %*% sqDist
      Sigma[iR, iC] <- t1^2*exp(-sqDist1/(2*t2^2))
    }
  }
  return(Sigma)
}


## k2 see Rasmussen 5.16
## adds some non-periodicity
calcK2Cov <- function(X1, X2, t3, t4, t5){
  Sigma <- matrix(0, nrow = nrow(X1), ncol = nrow(X2))
  for (iR in 1:nrow(X1)){
    for (iC in 1:nrow(X2)){
      sqDist <- X1[iR,] - X2[iC, ]
      sqDist1 <- t(sqDist) %*% sqDist
      Sigma[iR, iC] <- t3^2*exp(-sqDist1/(2*t4^2) - 
                                  (2*(sin(pi*sqrt(sqDist1)))^2)/(t5^2)
                                )
    }
  }
  return(Sigma)
}

## k3 see Rasmussen 5.17
## medium term irregularities
calcK3Cov <- function(X1, X2, t6, t7, t8){
  Sigma <- matrix(0, nrow = nrow(X1), ncol = nrow(X2))
  for (iR in 1:nrow(X1)){
    for (iC in 1:nrow(X2)){
      sqDist <- X1[iR,] - X2[iC, ]
      sqDist1 <- t(sqDist) %*% sqDist
      Sigma[iR, iC] <- (t6^2)*(1 + sqDist1/(2*t7*t8))^-t8
    }
  }
  return(Sigma)
}

## k4 see Rasmussen 5.18
## medium term irregularities
## delta doesn't seem to be defined in the text so we give is a default 1
calcK4Cov <- function(X1, X2, t9, t10, t11, delta=1){
  Sigma <- matrix(0, nrow = nrow(X1), ncol = nrow(X2))
  for (iR in 1:nrow(X1)){
    for (iC in 1:nrow(X2)){
      sqDist <- X1[iR,] - X2[iC, ]
      sqDist1 <- t(sqDist) %*% sqDist
      Sigma[iR, iC] <- (t9^2)*exp(-sqDist1/(2*t10^2)) + t11*delta
    }
  }
  return(Sigma)
}


kFunc <- function(X1, X2, tParams){
  k1 <- calcK1Cov(X1, X2, t1=tParams$t1, t2=tParams$t2)
  k2 <- calcK2Cov(X1, X2, t3=tParams$t3, t4=tParams$t4, t5=tParams$t5)
  k3 <- calcK3Cov(X1, X2, t6=tParams$t6, t7=tParams$t7, t8=tParams$t8)
  k4 <- calcK4Cov(X1, X2, t9=tParams$t9, t10=tParams$t10, t11=tParams$t11)
  
  k <- k1 + k2 + k3 + k4
  return(k)
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
posterior_pred <- function(x_new, x_train, y_train, tParams, sigma_n=1e-8){
  
  ## now we calculate the various k matrices
  k_xx   <- kFunc(x_train, x_train, tParams)
  k_xxs  <- kFunc(x_train, x_new,   tParams)
  k_xsx  <- kFunc(x_new,   x_train, tParams)
  k_xsxs <- kFunc(x_new,   x_new,   tParams)
  
  ## and the matrix inverses
  kInv <- solve(k_xx + sigma_n*diag(nrow(k_xx)))
  predMean <- k_xsx %*% kInv %*% y_train
  predCov <- k_xsxs - k_xsx %*% kInv %*% k_xxs
  
  ## the marginal likelihood
  marginal_llk <- -0.5*t(y_train) %*% kInv %*% y_train - 
    0.5*log(det(k_xx + sigma_n*diag(nrow(k_xx))) + 1e-10) - 
    nrow(x_train)/2*log(2*pi)
  
  output <- list(predMean = predMean, 
                 predCov = predCov,
                 marginal_llk = marginal_llk)
  return(output)
}


## do some gp
xNew <- as.matrix(testDat$dt_idx, ncol=1)
xTrain <- as.matrix(trainDat$dt_idx, ncol=1)
yTrain <- as.matrix(trainDat$co2_norm, ncol=1)
tParams <- lapply(1:11, function(x) 1)
names(tParams) <- paste0("t", 1:11)
x <- posterior_pred(x_new = xNew, x_train = xTrain, y_train = yTrain, tParams = tParams, 1e-8)

predDf <- data.table(x = xTrain,  y = yTrain, type = "TRAIN") %>% 
  rbind(data.table(x = xNew, y = x$predMean, type = "PRED")) %>% 
  rbind(data.table(x = xNew, y = testDat$co2_norm, type = "ACTUAL"), use.names=FALSE) %>% 
  setnames(., c("x", "y", "type"))

ggplot(predDf, aes(x = x, y = y)) + 
  geom_point(aes(colour = type), alpha = 0.5)

## stopped here: 
## - how to do the hyperparameter optimisation?
## - look at the training fit

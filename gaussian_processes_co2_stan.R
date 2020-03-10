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


## STAN (by me) ----------------------------------------------------------------
## see examples https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html#2_gaussian_process_regression
## other references:
## https://betanalpha.github.io/writing/
## https://github.com/betanalpha/knitr_case_studies/tree/master/gaussian_processes
## https://discourse.mc-stan.org/t/implementig-periodic-covariance-function/1787
## https://github.com/stan-dev/stan/wiki/Adding-a-Gaussian-Process-Covariance-Function
## https://github.com/stan-dev/stan/wiki/Adding-a-Gaussian-Process-Covariance-Function
## https://mc-stan.org/docs/2_19/functions-reference/covariance.html
## https://docs.pymc.io/notebooks/GP-MaunaLoa.html
## https://gitlab.com/hnickisch/gpml-matlab/-/tree/master/doc
## 
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores() - 2)

## run the stan file
stanData <- list(alpha=3,  # scale parameter of the cov kernel
                 rho=5.5,  # characteristic length param
                 sigma=2,  # noise variance
                 N=sum(cleanDat$isTrain), 
                 x=cleanDat$dt_idx[cleanDat$isTrain==TRUE])

## sample --> just generating some random samples 
simu_fit <- stan(file='gp_stan_gaussian_prior.stan',
                 data=stanData, iter=1,
                 chains=1, seed=494838, algorithm="Fixed_param")

f_total <- extract(simu_fit)$f[1,]   ## multivariate normal samples
y_total <- extract(simu_fit)$y[1,]   ## normal samples with mu = f and sigma from input
true_realization <- data.table(stanData$x,
                               cleanDat$co2_norm[cleanDat$isTrain],
                               cleanDat$co2[cleanDat$isTrain],
                               f_total, 
                               y_total)
setnames(true_realization, c("dt_idx", 
                             "co2_norm",
                             "co2",
                             "f_total", 
                             "y_total"))

ggplot(true_realization, aes(x = dt_idx, y = y_total)) + 
  geom_point() + 
  geom_line(aes(x = dt_idx, y = f_total)) + 
  geom_point(aes(x = dt_idx, y = co2_norm), colour = "red")


## now incorporate real data and predict out of sample data
pred_data <- list(alpha=3, 
                  rho=5.5, 
                  sigma=2, 
                  N=sum(cleanDat$isTrain), 
                  x=cleanDat$dt_idx[cleanDat$isTrain==TRUE],
                  y=cleanDat$co2[cleanDat$isTrain==TRUE],
                  N_predict=nrow(cleanDat[isTrain==FALSE]), 
                  x_predict=cleanDat$dt_idx[cleanDat$isTrain==FALSE])

pred_fit <- stan(file='gp_stan_predict_gauss.stan', 
                 data=pred_data, 
                 iter=1000,
                 warmup=0,
                 chains=1, 
                 seed=5838298, 
                 refresh=1000, 
                 algorithm="Fixed_param")


params <- extract(pred_fit)
predDf <- as.data.table(t(params$f_predict)) %>% 
  .[, dt_idx := cleanDat$dt_idx[cleanDat$isTrain == FALSE]] %>% 
  melt.data.table(id.vars = "dt_idx")

ggplot(predDf, aes(x = dt_idx, y = value)) + 
  geom_line(alpha = 0.5) +
  geom_point(data = cleanDat, aes(x = dt_idx, y = co2))
## model isn't great as it goes to zero due to only using the sq exp kernel
## converges to the mean value of 0


## using all the real data
pred_data_all <- list(alpha=3, 
                      rho=5.5, 
                      sigma=2, 
                      N=nrow(cleanDat), 
                      x=cleanDat$dt_idx,
                      y=cleanDat$co2,
                      N_predict=nrow(cleanDat), 
                      x_predict=cleanDat$dt_idx)

pred_fit_all <- stan(file='gp_stan_predict_gauss.stan', 
                     data=pred_data_all, 
                     iter=1000,
                     warmup=0,
                     chains=1, 
                     seed=5838298, 
                     refresh=1000, 
                     algorithm="Fixed_param")


params_all <- extract(pred_fit_all)
predDf_all <- as.data.table(t(params_all$f_predict)) %>% 
  .[, dt_idx := cleanDat$dt_idx] %>% 
  melt.data.table(id.vars = "dt_idx")

ggplot(predDf_all, aes(x = dt_idx, y = value)) + 
  geom_line(alpha = 0.5) +
  geom_point(data = cleanDat, aes(x = dt_idx, y = co2))

## paper on using bayesian for item response theory
## https://arxiv.org/pdf/1905.09501.pdf